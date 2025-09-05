"""
Author: Sayantan Ghosh (https://github.com/Lazy-Coder-03)
"""
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib

# Define the path where model artifacts are saved
MODEL_PATH = 'model/distilbert_model'
RESULTS_PATH = 'output/results'
os.makedirs(RESULTS_PATH, exist_ok=True)


class ReviewDataset(Dataset):
    """Custom Dataset for customer reviews."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model():
    """Performs model evaluation and generates reports."""
    try:
        # --- 1. Load Data, Model, Tokenizer, and LabelEncoder ---
        print("✅ Loading data and model artifacts...")
        df = pd.read_csv('data/sample_data_cleaned.csv')
        
        # Load the LabelEncoder to get correct category names
        le = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
        label_names = le.classes_

        X = df['query_clean'].tolist()
        y = le.transform(df['category'].tolist())

        # Use the same data split as in train.py
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Determine the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the fine-tuned model and tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)

        # --- 2. Tokenize and Prepare Data for Evaluation ---
        print("✅ Tokenizing test data...")
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)
        test_dataset = ReviewDataset(test_encodings, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # --- 3. Run Predictions ---
        print("✅ Running predictions on the test set...")
        model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
        
        logits = torch.cat(all_logits, dim=0)
        y_pred = np.argmax(logits.numpy(), axis=1)
        confidences = torch.softmax(logits, dim=1).max(dim=1).values.numpy()

        # --- 4. Calculate Metrics and Create Reports ---
        print("✅ Generating evaluation reports and visualizations...")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, target_names=label_names)

        # Generate F1 Score Plot
        f1_scores = [report[label]['f1-score'] for label in label_names]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=label_names, y=f1_scores, palette='viridis', hue=label_names, legend=False)
        plt.title('F1 Score per Category')
        plt.ylabel('F1 Score')
        plt.xlabel('Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.ylim(0, 1.1)
        plt.savefig(os.path.join(RESULTS_PATH, 'f1_scores.png'))
        plt.close()

        # Generate Confidence Distribution Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(confidences, bins=20, kde=True)
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'confidence_distribution.png'))
        plt.close()

        # Generate Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'))
        plt.close()

        # Create and save a JSON summary report
        category_metrics = {label: report[label] for label in label_names}
        summary = {
            'overall_accuracy': accuracy,
            'average_confidence': float(np.mean(confidences)),
            'test_cases_run': len(y_test),
            'category_f1_scores': {label: report[label]['f1-score'] for label in label_names},
            'category_metrics': category_metrics
        }
        with open(os.path.join(RESULTS_PATH, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print('✅ Evaluation complete. Summary, metrics, and visualizations saved to output/results/.')

    except FileNotFoundError as e:
        print(f'❌ Error: A required file was not found. Please ensure that {e.filename} exists.')
        print("Make sure you have run `train.py` first to create the necessary model files.")
    except Exception as e:
        print(f'❌ An unexpected error occurred during evaluation: {e}')

if __name__ == '__main__':
    evaluate_model()