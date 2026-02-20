"""
Upload the trained sentiment model to HuggingFace Hub.

Run this AFTER training is complete (models/sentiment-model/ must exist).

Steps:
  1. Get your token from https://huggingface.co/settings/tokens  (type: Write)
  2. Run: python upload_to_hub.py
  3. Paste your token when prompted
"""

import os
from huggingface_hub import HfApi, create_repo, login
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIGURATION — your HuggingFace username and repo name
# ============================================================
HF_USERNAME = "cringepnh"
REPO_NAME = "koelectra-korean-sentiment"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "sentiment-model")

FULL_REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"


def upload():
    print("🚀 Uploading model to HuggingFace Hub...")
    print(f"   Destination: https://huggingface.co/{FULL_REPO_ID}")
    print()

    # Step 1: Login with your HuggingFace token
    # Get your token at: https://huggingface.co/settings/tokens  (type: Write)
    token = input("🔑 Paste your HuggingFace token here: ").strip()
    login(token=token)
    print("   ✅ Logged in successfully.")
    print()

    # Step 2: Create the repository on HuggingFace (if it doesn't exist)
    print("📁 Creating repository (if not exists)...")
    create_repo(
        repo_id=FULL_REPO_ID,
        repo_type="model",
        exist_ok=True,  # Don't error if it already exists
    )
    print("   ✅ Repository ready.")

    # Step 3: Load the saved model and tokenizer
    print(f"\n📦 Loading model from: {MODEL_DIR}")
    if not os.path.exists(MODEL_DIR):
        print("❌ Model not found! Run python main.py first to train and save the model.")
        return

    # Step 4: Push model and tokenizer to the Hub
    print("\n⏳ Uploading model weights (this may take a few minutes)...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model.push_to_hub(FULL_REPO_ID)
    tokenizer.push_to_hub(FULL_REPO_ID)

    # Step 5: Upload the model card (README.md on HuggingFace)
    model_card_path = os.path.join(os.path.dirname(__file__), "hf_model_card.md")
    if os.path.exists(model_card_path):
        print("\n📝 Uploading model card...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo="README.md",
            repo_id=FULL_REPO_ID,
            repo_type="model",
        )
        print("   ✅ Model card uploaded.")

    print(f"\n✅ Done! Your model is now live at:")
    print(f"   https://huggingface.co/{FULL_REPO_ID}")
    print()
    print("   You can load it from anywhere with:")
    print(f'   model = AutoModelForSequenceClassification.from_pretrained("{FULL_REPO_ID}")')
    print(f'   tokenizer = AutoTokenizer.from_pretrained("{FULL_REPO_ID}")')


if __name__ == "__main__":
    upload()
