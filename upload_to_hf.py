from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="model/distilbert_model",
    repo_id="Lazycoder03/DistilBERT-Customer-Review-Classifier",
    repo_type="model",
)
