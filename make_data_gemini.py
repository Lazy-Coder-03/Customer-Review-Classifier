"""
Created as part of Liberty Infospace hiring test – not for production use without candidate's consent
Author: Sayantan Ghosh
GitHub: https://github.com/lazy-coder-03
"""

import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv() # This line loads the variables from .env
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

            # Use more specific prompts for each category to reduce ambiguity
            if category == "Billing Issue":
                prompt = f"""
                Write {batch_size} short, realistic, human-like customer reviews (1-2 sentences each).
                Each review must be a plausible billing issue for a business software product. Use specific terms like 'invoice', 'charged', 'refund', 'payment', 'fee', 'overcharged', 'billing cycle', or 'discrepancy'.
                Do NOT number or bullet them, just return raw reviews line by line.
                """
            elif category == "Technical Problem":
                prompt = f"""
                Write {batch_size} short, realistic, human-like customer reviews (1-2 sentences each).
                Each review must be a plausible technical problem with a business software product. Use specific technical terms like 'error code', 'bug', 'glitch', 'crashed', 'API', 'integration failure', 'slow performance', or 'unresponsive'.
                Do NOT number or bullet them, just return raw reviews line by line.
                """
            elif category == "Compliment":
                prompt = f"""
                Write {batch_size} short, realistic, human-like customer reviews (1-2 sentences each).
                Each review must be a compliment about a business software product. Focus on positive aspects like 'intuitive', 'easy to use', 'great support', 'time-saver', or 'streamlined workflow'.
                Do NOT number or bullet them, just return raw reviews line by line.
                """
            elif category == "Product Question":
                prompt = f"""
                Write {batch_size} short, realistic, human-like customer reviews (1-2 sentences each).
                Each review must be a question about a business software product's functionality, features, or how to perform a task. Use question marks and phrases like 'how do I', 'is there a way to', or 'can I'.
                Do NOT number or bullet them, just return raw reviews line by line.
                """
            elif category == "Complaint":
                prompt = f"""
                Write {batch_size} short, realistic, human-like customer reviews (1-2 sentences each).
                Each review must be a general complaint or expression of frustration about a business software product. Avoid specific technical or billing terms. Use general negative phrases like 'unacceptable', 'frustrating', 'disappointed', 'terrible', or 'waste of money'.
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

print(f"✅ Generated dataset with {len(df)} samples saved to data/sample_data.csv")
print(df.head(10))