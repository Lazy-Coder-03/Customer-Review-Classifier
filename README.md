# 🤖 Customer Review Classifier App

### An end-to-end solution for multi-class text classification using NLP and machine learning.

This project is a complete and ready-to-use application that classifies customer reviews into predefined categories. It leverages a fine-tuned DistilBERT model for high-accuracy predictions and features a user-friendly Streamlit frontend. The entire application is designed for seamless deployment on the Hugging Face Hub.

## ✨ Key Features

* **Data Preprocessing**: Scripts to clean and prepare raw customer review data.

* **Model Training**: A robust training pipeline for fine-tuning a **DistilBERT** model for multi-class classification.

* **Model Hosting**: The trained model is hosted on the **Hugging Face Hub**, enabling direct use without local storage.

* **Comprehensive Evaluation**: The `evaluate.py` script generates a full suite of metrics, including a **confusion matrix** and visualizations of model performance.

* **Streamlit App**: A single-file, production-ready Streamlit application that handles both the frontend and backend logic.

* **Professional UI**: The app includes a responsive design, clear metric explanations, and professional-looking charts.

## ⚙️ How It Works

### Project Workflow

1. **Data Preparation**: The `prepare_data.py` script cleans and preprocesses the raw data from `data/sample_data.csv`.

2. **Model Training**: The `train.py` script fine-tunes the DistilBERT model. The output model and tokenizer are saved locally.

3. **Model Upload**: The `upload_to_hf.py` script pushes the trained model and tokenizer to the Hugging Face Hub, making it accessible to the app.

4. **Model Evaluation**: The `evaluate.py` script evaluates the model's performance on a test set and saves the results (`summary_report.json`, `confusion_matrix.png`, etc.) to `output/results/`.

5. **App Deployment**: The `app.py` script serves as the main application. It loads the model directly from the Hugging Face Hub and provides a user interface for classification and metric visualization.

### Architecture

The application uses a simplified architecture where the Streamlit frontend directly loads and runs the model. This eliminates the need for a separate FastAPI backend and is the standard practice for deploying a single-model app on Hugging Face Spaces.

## 📂 File Structure

```

Customer-Review-Classifier/
├── data/
│   ├── sample\_data.csv           \# Raw customer review data
│   └── sample\_data\_cleaned.csv   \# Cleaned and encoded data
├── output/
│   ├── distilbert\_model/         \# Local copy of the trained model (optional)
│   ├── results/
│   │   ├── summary\_report.json   \# JSON report of all evaluation metrics
│   │   ├── f1\_scores.png         \# F1 scores visualization
│   │   ├── confidence\_distribution.png \# Confidence distribution visualization
│   │   └── confusion\_matrix.png  \# Visual representation of model confusion
├── src/
│   ├── prepare\_data.py           \# Script for data cleaning and encoding
│   ├── train.py                  \# Script for fine-tuning the model
│   ├── evaluate.py               \# Script for model evaluation and reporting
│   ├── app.py                    \# The main Streamlit application
│   └── upload\_to\_hf.py           \# Script to push the model to the Hugging Face Hub
├── requirements.txt              \# Project dependencies
├── .gitignore                    \# Specifies files to ignore in Git (e.g., large models)
└── README.md

```

## 🚀 Quickstart: Running Locally

1. **Clone the Repository**

```

git clone [https://github.com/Lazy-Coder-03/Customer-Review-Classifier.git](https://www.google.com/search?q=https://github.com/Lazy-Coder-03/Customer-Review-Classifier.git)
cd Customer-Review-Classifier

```

2. **Install Dependencies**

```

pip install -r requirements.txt

```

3. **Prepare the Data**

```

python src/prepare\_data.py

```

4. **Train the Model**

```

python src/train.py

```

5. **Evaluate the Model**

```

python src/evaluate.py

```

This step generates the metrics and visualizations for the app dashboard.

6. **Run the App**

```

streamlit run src/app.py

```

Access the app in your browser at [http://localhost:8501](http://localhost:8501).

## ☁️ Deployment on Hugging Face Spaces

This project is configured for seamless deployment on Hugging Face Spaces by linking your GitHub repository. Any changes pushed to the `main` branch will automatically trigger a new build.

1. **Upload Your Model to the Hub**
Your model's files must be on the Hub. Use your local scripts to push the model.

```

# Log in to your Hugging Face account

huggingface-cli login

# Push the local model files to the Hub

python src/upload\_to\_hf.py

```

2. **Link Your GitHub Repo to a Space**

* Create a new Space on [huggingface.co/new](https://huggingface.co/new).

* Select **Streamlit** as the SDK.

* Choose your `Customer-Review-Classifier` GitHub repository from the dropdown.

The Space will automatically build and deploy your app. The `app.py` script will handle downloading the model from the Hugging Face Hub when it first runs.

## ✍️ Author

[Sayantan Ghosh](https://github.com/Lazy-Coder-03)

---
