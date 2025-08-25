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

# Define the Hugging Face Hub model ID
model_id = "Lazycoder03/DistilBERT-Customer-Review-Classifier"

# --- Streamlit App UI ---
# Use the "centered" layout for a cleaner, more professional look
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


if 'page_loaded' not in st.session_state:
    with st.spinner("Loading page..."):
        st.session_state['page_loaded'] = True
st.title('Customer Review Classifier')
st.markdown("<h3 style='text-align: center;'>Predict the category of a customer review using a fine-tuned DistilBERT model.</h3>", unsafe_allow_html=True)
st.markdown("---")

# Section to display model loading status
st.subheader("Model Loading Status")
info_message = st.info("Model and tokenizer are loading... This may take a few moments on the first run.")
try:
    with st.spinner("Downloading model from Hugging Face Hub..."):
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
        model = DistilBertForSequenceClassification.from_pretrained(model_id)
        model.eval()
    info_message.empty()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model from Hugging Face Hub: {e}")
    st.stop()
st.markdown("---")

# --- Other UI Elements ---
# Load category mapping and metrics
try:
    df = pd.read_csv('data/sample_data_cleaned.csv')
    cat_map_df = df[['category_encoded', 'category']].drop_duplicates().sort_values('category_encoded')
    idx_to_cat = dict(zip(cat_map_df['category_encoded'], cat_map_df['category']))
    cat_map = {idx: name for idx, name in idx_to_cat.items()}
except FileNotFoundError:
    st.error("Missing required file: `data/sample_data_cleaned.csv`. Please upload it to your Space.")
    st.stop()

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
                cat_metrics_df.index = [cat_map.get(int(k), k) for k in cat_metrics_df.index]
                st.table(cat_metrics_df)

                fig, ax = plt.subplots(figsize=(10, 6))
                cat_metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax, title='Precision, Recall, and F1-Score by Category')
                ax.set_ylabel('Score')
                ax.set_xlabel('Category')
                ax.set_ylim(0, 1.1)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                cat_metrics_df['support'].plot(kind='bar', ax=ax2, color='skyblue', title='Sample Support by Category')
                ax2.set_ylabel('Number of Samples')
                ax2.set_xlabel('Category')
                ax2.tick_params(axis='x', rotation=45)
                st.pyplot(fig2)
        
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
query = st.text_area('Enter a customer review:', key='review_input')
if st.button('Classify Text', key='classify_btn'):
    if query.strip():
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
        
        conf_labels = [cat_map.get(i, str(i)) for i in range(len(confidences))]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.barh(conf_labels, np.array(confidences)*100, color='#6fa8dc')
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Confidence by Category')
        st.pyplot(fig)
        
    else:
        st.warning('Please enter some text.')
        
st.markdown("---")
st.markdown("##### About this App")
st.markdown("""
This app uses a fine-tuned DistilBERT model to classify customer reviews into predefined categories. The model was trained on a dataset of customer queries and their corresponding categories. by [Sayantan Ghosh](https://github.com/Lazy-Coder-03)
""")