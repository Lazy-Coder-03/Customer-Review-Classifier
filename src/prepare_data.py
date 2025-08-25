import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
# Load data
df = pd.read_csv('data/sample_data.csv')

# Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(subset=['query', 'category'], inplace=True)

# Clean text: lowercase, remove extra spaces, remove special chars
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['query_clean'] = df['query'].apply(clean_text)

# Categorize: ensure categories are consistent
df['category'] = df['category'].str.strip().str.title()

# Encode categories for classification

le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# Save cleaned data for modeling
df.to_csv('data/sample_data_cleaned.csv', index=False)

print('✅ Data loaded, cleaned, and categorized. Ready for classification.')
print(df.head())
