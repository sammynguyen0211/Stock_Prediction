import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from imblearn.pipeline import Pipeline
import shap

warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint_bitcoin = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)
df_prices = get_bitcoin_historical_prices()

MIN_VAL = float(0.5 * df_prices.iloc[:, 0].min())
MAX_VAL = float(2.0 * df_prices.iloc[:, 0].max())
DEFAULT_VAL = float(df_prices.iloc[:, 0].mean())

MODEL_INFO = {
    "endpoint": aws_endpoint_bitcoin,
    "explainer": 'explainer_bitcoin.shap',
    "pipeline": 'finalized_bitcoin_model.tar.gz',
    "keys": ["Close Price (USD)"],
    "inputs": [{"name": "Close Price (USD)", "min": MIN_VAL, "max": MAX_VAL, "default": DEFAULT_VAL, "step": 100.0}]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(Filename=filename, Bucket=bucket, Key=f"{key}/{os.path.basename(filename)}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(int(pred_val), str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

def display_signal_plot(res):
    fig, ax = plt.subplots(figsize=(10, 4))
    prices = df_prices.iloc[-30:, 0].values
    if res == "BUY":
        ax.plot(prices, color='green', linewidth=2)
        ax.set_title("BUY Signal — Upward Trend Expected", color='green', fontsize=14)
        st.success("📈 BUY Signal Detected!")
    elif res == "SELL":
        ax.plot(prices, color='red', linewidth=2)
        ax.set_title("SELL Signal — Downward Trend Expected", color='red', fontsize=14)
        st.error("📉 SELL Signal Detected!")
    else:
        ax.plot(prices, color='orange', linewidth=2)
        ax.set_title("HOLD Signal — Sideways Trend Expected", color='orange', fontsize=14)
        st.warning("⏸ HOLD Signal Detected!")
    ax.set_xlabel("Days")
    ax.set_ylabel("Bitcoin Close Price (USD)")
    st.pyplot(fig)

def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )
    full_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])
    input_df_transformed = preprocessing_pipeline.transform(input_df)
    shap_values = explainer(input_df_transformed)
    feature_names = full_pipeline.named_steps['feature_selection'].get_feature_names_out()

    exp = shap.Explanation(
        values=shap_values[0, :, 0],
        base_values=explainer.expected_value[0],
        data=input_df_transformed[0],
        feature_names=feature_names
    )
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(exp)
    st.pyplot(fig)
    top_feature = pd.Series(exp.values, index=exp.feature_names).abs().idxmax()
    st.info(f"**Most influential factor:** {top_feature}")

# UI
st.set_page_config(page_title="Bitcoin Signal Predictor", layout="wide")
st.title("🪙 Bitcoin BUY / HOLD / SELL Predictor")
st.markdown("Enter a Bitcoin closing price to predict whether you should **Buy**, **Hold**, or **Sell**.")

with st.form("pred_form"):
    st.subheader("Input")
    user_inputs = {}
    for inp in MODEL_INFO["inputs"]:
        user_inputs[inp['name']] = st.number_input(
            inp['name'],
            min_value=inp['min'],
            max_value=inp['max'],
            value=inp['default'],
            step=inp['step']
        )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    new_row = pd.DataFrame([[user_inputs["Close Price (USD)"]]], columns=["Close Price (USD)"])
    input_df = pd.concat([df_prices, new_row], ignore_index=True)
    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction", res)
        display_signal_plot(res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
