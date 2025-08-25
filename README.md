# Customer Review Classification App

## Overview
This project is an end-to-end solution for classifying customer reviews into categories using NLP and machine learning. It features:
- Data cleaning and preprocessing
- Model training with DistilBERT
- Evaluation and visualization
- FastAPI backend for predictions
- Streamlit frontend for user interaction
- Model hosting on Hugging Face Hub

## How It Works

### Data Preparation
- Run `prepare_data.py` to clean and encode your raw data. Output: `data/sample_data_cleaned.csv`.

### Model Training
- Run `train.py` to train a DistilBERT classifier on the cleaned data. The script saves the trained model and tokenizer to `output/distilbert_model/`.

### Model Evaluation
- Run `evaluate.py` to:
  - Test the trained model on the test set
  - Calculate overall accuracy, average confidence, and category-wise F1 scores
  - Generate visualizations (F1 score per category, confidence distribution)
  - Output a summary report in `output/results/summary_report.json`


### App Usage
- The Streamlit app (`app.py`) provides:
   - Model loading status and metrics dashboard
   - Input for customer review text
   - Prediction card with category and confidence
   - Horizontal bar chart of confidence scores for all categories
   - Metrics and visualizations from evaluation

**No separate backend is required. All logic runs in Streamlit.**


### Model Hosting
- The trained model is already hosted at: [https://huggingface.co/Lazycoder03/DistilBERT-Customer-Review-Classifier](https://huggingface.co/Lazycoder03/DistilBERT-Customer-Review-Classifier)
- You can use this hosted model directly in your app and API without needing to retrain or upload.
- To use the model locally, download and place it in `output/distilbert_model/` (optional).

## File Structure
```text
AI_ML/
├── data/
│   ├── sample_data.csv
│   └── sample_data_cleaned.csv
├── output/
│   ├── distilbert_model/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   │   ├── special_tokens_map.json
│   │   └── ...
│   ├── results/
│   │   ├── summary_report.json
│   │   ├── f1_scores.png
│   │   └── confidence_distribution.png
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── api.py
│   ├── app.py
│   ├── main.py
│   └── upload_to_hf.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Quickstart
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   ```sh
   python src/prepare_data.py
   ```
3. **Train model:**
   ```sh
   python src/train.py
   ```
4. **Evaluate model:**
   ```sh
   python src/evaluate.py
   ```
5. **Upload model to Hugging Face Hub:**
   ```sh
   huggingface-cli login
   python src/upload_to_hf.py
   ```
6. **Start the app:**
   ```sh
   python src/main.py
   ```

7. **Access the app:**
   - Open [http://localhost:8501](http://localhost:8501) in your browser after running the command above.

## Notes
- Large model files are not tracked in Git. See `.gitignore`.
- For custom categories, update `cat_map` in `app.py` and `api.py`.
- For issues or improvements, open an issue or pull request.

---
**Author:** Sayantan Ghosh ([Lazycoder03](https://github.com/lazy-coder-03))
