import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load test data and model
try:
    df = pd.read_csv('data/sample_data_cleaned.csv')
    X = df['query_clean'].tolist()
    y = df['category_encoded'].tolist()
    # Use same split as train.py
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tokenizer = DistilBertTokenizerFast.from_pretrained('model/distilbert_model')
    model = DistilBertForSequenceClassification.from_pretrained('model/distilbert_model')
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

    class ReviewDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    test_dataset = ReviewDataset(test_encodings, y_test)
    # Run test cases with DataLoader
    from torch.utils.data import DataLoader
    model.eval()
    all_logits = []
    with torch.no_grad():
        loader = DataLoader(test_dataset, batch_size=32)
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    y_pred = np.argmax(logits.numpy(), axis=1)
    confidences = torch.softmax(logits, dim=1).max(dim=1).values.numpy()

    # Calculate accuracy & category-wise performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Visualizations
    labels = list(report.keys())[:-3]
    f1_scores = [report[label]['f1-score'] for label in labels]
    # Map indices to category names
    cat_map = {
        '0': 'Billing Issue',
        '1': 'Technical Problem',
        '2': 'Compliment',
        '3': 'Product Question',
        '4': 'Complaint'
    }
    label_names = [cat_map.get(label, label) for label in labels]
    plt.figure(figsize=(8,4))
    sns.barplot(x=label_names, y=f1_scores)
    plt.title('F1 Score per Category')
    plt.ylabel('F1 Score')
    plt.xlabel('Category')
    plt.tight_layout()
    plt.savefig('output/f1_scores.png')
    plt.close()

    plt.figure(figsize=(8,4))
    sns.histplot(confidences, bins=20, kde=True)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('output/results/confidence_distribution.png')
    plt.close()

    # Output summary report
    summary = {
        'overall_accuracy': accuracy,
        'average_confidence': float(np.mean(confidences)),
        'test_cases_run': len(y_test),
        'category_f1_scores': dict(zip(labels, f1_scores))
    }
    with open('output/results/summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('✅ Evaluation complete. Summary, metrics, and visualizations saved to output/.')
except Exception as e:
    print(f'❌ Error during evaluation: {e}')
