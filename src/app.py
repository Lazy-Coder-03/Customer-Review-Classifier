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

# Define the Hugging Face Hub model ID
model_id = "Lazycoder03/DistilBERT-Customer-Review-Classifier"

# --- Streamlit App UI ---
st.title('Customer Review Classifier')

# Section to display model loading status
st.subheader("Model Loading Status")
info_message = st.info("Model and tokenizer are loading... This may take a few moments on the first run.")

try:
    # Use st.spinner to show a professional-looking "loading" state
    with st.spinner("Downloading from Hugging Face Hub..."):
        # Load the model and tokenizer
        # The transformers library handles downloading and caching
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_id)
        model = DistilBertForSequenceClassification.from_pretrained(model_id)
        model = model.to('cpu')
        model.eval()

    # Clear the info message once the model is loaded
    info_message.empty()
    st.success("✅ Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model from Hugging Face Hub: {e}")
    st.stop()

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
f1_img_path = 'output/results/f1_scores.png'
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

# Display metrics if available
if metrics:
    st.subheader('Model Metrics & Performance')
    st.markdown("""
                
    **Accuracy**: The percentage of correct predictions.
    
    **Precision**: Of all predicted instances of a category, how many were correct.
    
    **Recall**: Of all actual instances of a category, how many were found.
    
    **F1 Score**: A balance of precision and recall.
    
    """)
    st.write(f"**Overall Accuracy:** `{metrics['accuracy']:.2f}`")
    st.write('**Category-wise F1 Scores:**')
    f1_scores = {cat_map.get(int(k), k): v for k, v in metrics['category_f1_scores'].items() if k.isdigit()}
    st.table(f1_scores)
    if os.path.exists(f1_img_path):
        st.image(f1_img_path, caption='F1 Score per Category', use_container_width=True)
else:
    st.info('Model metrics not found. Please upload `output/results/` to your repository.')

# --- Inference Section ---
st.subheader("Predict Customer Review")
query = st.text_area('Enter a customer review:', key='review_input')
if st.button('Classify Text', key='classify_btn'):
    if query.strip():
        # Perform inference
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=128)
        for k in inputs:
            inputs[k] = inputs[k].to('cpu')
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
            confidences = probs.tolist()
            
        category = idx_to_cat.get(pred_idx, 'Unknown')
        
        # Display prediction and confidence chart
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
        st.markdown("---")
        st.markdown("##### About this App")
        st.markdown("""
        This app uses a fine-tuned DistilBERT model to classify customer reviews into predefined categories. The model was trained on a dataset of customer queries and their corresponding categories. by [Sayantan Ghosh](https://github.com/Lazy-Coder-03)
        """)
        
    else:
        st.warning('Please enter some text.')