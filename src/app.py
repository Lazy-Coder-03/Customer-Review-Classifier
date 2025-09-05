"""
Author: Sayantan Ghosh (https://github.com/Lazy-Coder-03)
"""
import streamlit as st
import json
import os
import numpy as np
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
import io
import seaborn as sns
from PIL import Image
import joblib

# --- Developer-facing configuration ---
# Set to True to load the model from a local directory.
# Set to False to download the model from the Hugging Face Hub.
USE_LOCAL_MODEL = True

# Define paths
MODEL_PATH = 'model/distilbert_model'
HUB_MODEL_ID = "Lazycoder03/DistilBERT-Customer-Review-Classifier"
LOCAL_LABEL_ENCODER_PATH = os.path.join(MODEL_PATH, 'label_encoder.pkl')

# --- Model Loading with Caching ---
@st.cache_resource
def load_model_and_tokenizer():
    """Loads the model, tokenizer, and label encoder based on the configuration flag."""
    if USE_LOCAL_MODEL:
        try:
            st.info("Using local model...")
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
            model.eval()

            # Load the LabelEncoder from the local directory
            le = joblib.load(LOCAL_LABEL_ENCODER_PATH)
            idx_to_cat = {i: name for i, name in enumerate(le.classes_)}
            return tokenizer, model, idx_to_cat
        except FileNotFoundError:
            st.error(f"Error: Local model files not found in '{MODEL_PATH}'. Please train the model first or switch to download mode.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading local model: {e}")
            st.stop()
    else:
        try:
            st.info(f"Downloading model from Hugging Face Hub: {HUB_MODEL_ID}")
            tokenizer = DistilBertTokenizerFast.from_pretrained(HUB_MODEL_ID)
            model = DistilBertForSequenceClassification.from_pretrained(HUB_MODEL_ID)
            model.eval()

            # Load label encoder (assuming it's in the repo or a local file)
            if os.path.exists(LOCAL_LABEL_ENCODER_PATH):
                le = joblib.load(LOCAL_LABEL_ENCODER_PATH)
                idx_to_cat = {i: name for i, name in enumerate(le.classes_)}
            else:
                st.warning("Warning: LabelEncoder not found. Using hardcoded labels as a fallback.")
                idx_to_cat = {0: 'Billing Issue', 1: 'Complaint', 2: 'Compliment', 3: 'Product Question', 4: 'Technical Problem'}
            return tokenizer, model, idx_to_cat
        except Exception as e:
            st.error(f"Error downloading model from Hugging Face Hub: {e}")
            st.stop()

# --- Streamlit App UI ---
st.set_page_config(page_title="Customer Review Classifier", page_icon="🤖", layout="centered")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .reportview-container .main .block-container{
        max-width: 800px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .css-1y4pm5l {
        padding-top: 0rem;
    }
    h1, h2, h3 {
        text-align: center;
    }
    .stTable {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title('Customer Review Classifier')
st.markdown("<h3 style='text-align: center;'>Predict the category of a customer review using a fine-tuned DistilBERT model.</h3>", unsafe_allow_html=True)
st.markdown("---")

# Section to display model loading status
st.subheader("Model Loading Status")
with st.spinner("Loading model..."):
    tokenizer, model, idx_to_cat = load_model_and_tokenizer()
    st.success("✅ Model loaded successfully!")

st.markdown("---")

# --- Other UI Elements ---
metrics = None
metrics_path = 'output/results/summary_report.json'
confusion_matrix_path = 'output/results/confusion_matrix.png'

if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if metrics:
        st.subheader('Model Metrics & Performance')
        st.markdown("""
        **Accuracy**: The percentage of correct predictions out of all predictions.
        
        **Precision**: Of all predicted instances of a category, how many were actually correct.
        
        **Recall**: Of all actual instances of a category, how many were correctly found by the model.
        
        **F1 Score**: The harmonic mean of Precision and Recall, providing a single score that balances both.

        **Support**: The number of actual instances of a category in the test dataset.
        """)
        st.write(f"**Overall Accuracy:** `{metrics.get('accuracy', metrics.get('overall_accuracy', 0)):.2f}`")

        metrics_tab, confusion_tab = st.tabs(["Performance by Category", "Confusion Matrix"])

        with metrics_tab:
            st.markdown("#### Category-wise Metrics")
            if 'category_metrics' in metrics:
                cat_metrics_df = pd.DataFrame(metrics['category_metrics']).T
                # The keys in the new summary report are already the category names (strings),
                # so no need to map them. We simply display the table as is.
                st.table(cat_metrics_df)

                fig, ax = plt.subplots(figsize=(10, 6))
                cat_metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax, title='Precision, Recall, and F1-Score by Category')
                ax.set_ylabel('Score')
                ax.set_xlabel('Category')
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
        
        with confusion_tab:
            st.markdown("#### Confusion Matrix")
            if os.path.exists(confusion_matrix_path):
                image = Image.open(confusion_matrix_path)
                st.image(image, caption='Confusion Matrix', use_container_width=True)
            else:
                st.info('Confusion matrix not found. Please run `evaluate.py` to generate it.')
    else:
        st.info('Model metrics not found. Please upload `output/results/` to your repository.')
st.markdown("---")

# --- Inference Section ---
st.subheader("Predict Customer Review")
query = st.text_area('Enter a customer review:', key='review_input', placeholder="E.g., 'The product stopped working after a week of use.'", height=100)

# Check if text area is empty to enable/disable button
is_query_empty = not query.strip()
if st.button('Classify Text', key='classify_btn', disabled=is_query_empty):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        confidences = probs.tolist()
        
    category = idx_to_cat.get(pred_idx, 'Unknown')
    
    st.markdown(f"""
    <div style='background-color:#262730;color:#fafafa;padding:1em;border-radius:10px;margin-bottom:1em;'>
        <h4 style='color:#fafafa;'>Prediction</h4>
        <b>Category:</b> {category}<br>
        <b>Confidence:</b> {confidence*100:.2f}%
    </div>
    """, unsafe_allow_html=True)
    
    # Create the confidence plot using the dynamic category names
    conf_labels = [idx_to_cat.get(i, str(i)) for i in range(len(confidences))]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.barh(conf_labels, np.array(confidences)*100, color='#6fa8dc')
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Confidence by Category')
    st.pyplot(fig)
        
st.markdown("---")
st.markdown("##### About this App")
st.markdown("""
This app uses a fine-tuned DistilBERT model to classify customer reviews into predefined categories. The model was trained on a dataset of customer queries and their corresponding categories. by [Sayantan Ghosh](https://github.com/Lazy-Coder-03)
""")