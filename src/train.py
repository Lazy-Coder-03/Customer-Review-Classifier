"""
Author: Sayantan Ghosh (https://github.com/Lazy-Coder-03)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from transformers import EarlyStoppingCallback

# --- 1. Data Loading and Preparation ---
print("✅ Loading and preparing data...")
df = pd.read_csv('data/sample_data_cleaned.csv')

# Use the 'category' column to get the unique labels and fit the LabelEncoder
# This ensures a consistent mapping for all scripts
le = LabelEncoder()
le.fit(df['category'].unique())

X = df['query_clean'].tolist()
y = le.transform(df['category'].tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Model and Tokenizer Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
num_labels = len(le.classes_)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
model.to(device)

# --- 3. Tokenize Data ---
print("✅ Tokenizing data...")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, y_train)
test_dataset = ReviewDataset(test_encodings, y_test)

# --- 4. Training Configuration ---
print("✅ Setting up training arguments...")
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
training_args = TrainingArguments(
    output_dir='output/results/distilbert_results',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='output/logs/distilbert_logs',
    logging_steps=50,
    report_to=[],
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=callbacks
)

# --- 5. Training and Saving ---
print("✅ Starting model training...")
try:
    trainer.train()
    
    # Create model directory if it doesn't exist
    model_output_dir = 'model/distilbert_model'
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Save the fine-tuned model, tokenizer, and label encoder
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    joblib.dump(le, os.path.join(model_output_dir, 'label_encoder.pkl'))
    
    print(f'✅ DistilBERT model, tokenizer, and LabelEncoder saved to {model_output_dir}.')
except Exception as e:
    print(f'❌ Error during training: {e}')

# plt.tight_layout()
# plt.savefig('output/f1_scores.png')
# plt.close()