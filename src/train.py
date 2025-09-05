"""
Author: Sayantan Ghosh (https://github.com/Lazy-Coder-03)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import EarlyStoppingCallback

# Load cleaned data
df = pd.read_csv('data/sample_data_cleaned.csv')

# Features and labels
X = df['query_clean'].tolist()
y = df['category_encoded'].tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
num_labels = len(set(y))
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
model.to(device)

# Tokenize
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

import numpy as np
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

callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
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
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	callbacks=callbacks
)

trainer.train()

try:
	trainer.train()
	# Save model and tokenizer
	model.save_pretrained('model/distilbert_model')
	tokenizer.save_pretrained('model/distilbert_model')
	print('✅ DistilBERT model saved to model/.')
except Exception as e:
	print(f'❌ Error during training: {e}')
plt.tight_layout()
plt.savefig('output/f1_scores.png')
plt.close()

# Save model and tokenizer
model.save_pretrained('model/distilbert_model')
tokenizer.save_pretrained('model/distilbert_model')
print('✅ DistilBERT model, logs, metrics, and visualizations saved to output/.')
