"""
Author: Sayantan Ghosh (https://github.com/Lazy-Coder-03)
"""
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import string

# Load data
df = pd.read_csv('data/sample_data.csv')

# Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(subset=['query', 'category'], inplace=True)

# Define a more selective cleaning function
def clean_text_improved(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Remove only punctuation that is unlikely to be meaningful in this context
    # Keep symbols like #, $, !, ?
    text = re.sub(r'[^a-z0-9\s#$?!.]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Apply the new text cleaning and feature engineering
df['query_clean'] = df['query'].apply(clean_text_improved)

# Categorize: ensure categories are consistent
df['category'] = df['category'].str.strip().str.title()

# Encode categories for classification
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Save cleaned data for modeling
df.to_csv('data/sample_data_cleaned.csv', index=False)

print('✅ Data loaded, cleaned, and categorized. Ready for classification.')
print(df.head())