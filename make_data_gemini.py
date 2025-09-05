"""
Created as part of Liberty Infospace hiring test – not for production use without candidate's consent
Author: Sayantan Ghosh
GitHub: https://github.com/lazy-coder-03
"""

import google.generativeai as genai
import pandas as pd
import os

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

categories = ["Billing Issue", "Technical Problem", "Compliment", "Product Question", "Complaint"]

def generate_reviews(category, num=1000):
    """Generate realistic customer reviews for a given category using Gemini"""
    reviews = []
    batch_size = 200
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        for i in range(0, num, batch_size):
            print(f"Generating reviews {i+1} to {min(i+batch_size, num)} for category '{category}'...")
            prompt = f"""
            Write {batch_size} short, realistic, human-like customer reviews (1–2 sentences each).
            Each review should be plausible for a business software product and fall under the category: {category}.
            Do NOT number or bullet them, just return raw reviews line by line.
            """
            response = model.generate_content(prompt)
            batch_reviews = response.text.strip().split("\n")
            reviews.extend([{"query": r.strip(), "category": category} for r in batch_reviews if r.strip()])
        return reviews[:num]
    except Exception as e:
        print(f"Error generating reviews for category '{category}': {e}")
        return []

all_reviews = []
for cat in categories:
    all_reviews.extend(generate_reviews(cat, num=1000))

# Save to CSV
df = pd.DataFrame(all_reviews)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_data.csv", index=False)

print(f"✅ Generated dataset with {len(df)} samples saved to data/train_data.csv")
print(df.head(10))
