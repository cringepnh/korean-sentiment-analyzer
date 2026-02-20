"""
Korean Movie Review Sentiment Analyzer
========================================
This script fine-tunes a pretrained Korean language model (KoELECTRA)
on the NSMC dataset (200k Korean movie reviews) to classify reviews
as positive or negative.

Steps:
  1. Install dependencies (done via requirements.txt)
  2. Load the NSMC dataset from local TSV files (downloaded from github.com/e9t/nsmc)
  3. Clean and explore the data
  4. Tokenize reviews using the pretrained model's tokenizer
  5. Fine-tune the model on the training data
  6. Evaluate performance (accuracy, precision, recall, F1, confusion matrix)
  7. Test with custom Korean reviews
  8. Save the trained model for reuse
"""

import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================
# Set this to True to train on the FULL dataset (~150k samples).
# Set to False to use a small 5,000-sample subset for quick testing.
FULL_TRAINING = False

# How many samples to use for quick testing
QUICK_SAMPLE_SIZE = 5000

# The pretrained Korean model we'll fine-tune.
# KoELECTRA is a Korean ELECTRA model that understands Korean text well.
MODEL_NAME = "monologg/koelectra-base-finetuned-sentiment"

# Where to save the trained model
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "models", "sentiment-model")


# ============================================================
# STEP 2 — LOAD THE NSMC DATASET
# ============================================================
# The NSMC (Naver Sentiment Movie Corpus) is a famous Korean NLP dataset.
# It contains 200,000 Korean movie reviews:
#   - 150,000 for training
#   -  50,000 for testing
# Each review has a label: 0 = negative, 1 = positive
#
# The data files are tab-separated (TSV) with columns: id, document, label
# We downloaded them from: https://github.com/e9t/nsmc
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data():
    """Load the NSMC dataset from local TSV files."""
    print("=" * 60)
    print("STEP 2 — LOADING THE NSMC DATASET")
    print("=" * 60)

    # Read the tab-separated files using pandas.
    # These files have columns: id, document (the review text), label (0 or 1)
    train_path = os.path.join(DATA_DIR, "ratings_train.txt")
    test_path = os.path.join(DATA_DIR, "ratings_test.txt")

    print(f"\n📂 Loading training data from: {train_path}")
    print(f"📂 Loading test data from:     {test_path}")

    # read_csv with sep='\t' reads tab-separated files.
    # DataFrames are like spreadsheets — they let us view and manipulate data easily.
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    # Show the first 10 reviews so we can see what the data looks like
    print("\n📋 First 10 training reviews:")
    print(train_df.head(10).to_string())

    # Show how many reviews we have
    print(f"\n📊 Total training reviews: {len(train_df):,}")
    print(f"📊 Total test reviews:     {len(test_df):,}")

    # Show the label distribution — ideally we want a balanced dataset
    # (roughly equal numbers of positive and negative reviews)
    print("\n📊 Label distribution (0=negative, 1=positive):")
    print(train_df["label"].value_counts())

    return train_df, test_df


# ============================================================
# STEP 3 — CLEAN AND EXPLORE THE DATA
# ============================================================
# Real-world data is messy! We need to:
#   - Remove missing values (NaN) — rows where the review text is blank
#   - Remove duplicates — same review appearing multiple times
#   - Remove empty strings — reviews that exist but contain no text
def clean_data(train_df, test_df):
    """Clean the data by removing missing values, duplicates, and empty reviews."""
    print("\n" + "=" * 60)
    print("STEP 3 — CLEANING AND EXPLORING THE DATA")
    print("=" * 60)

    # --- Check for problems BEFORE cleaning ---
    print("\n🔍 Before cleaning:")
    print(f"  Training set size: {len(train_df):,}")
    print(f"  Test set size:     {len(test_df):,}")

    # Check for missing values (NaN = "Not a Number", means the cell is empty)
    print(f"\n  Missing values in training set:")
    print(f"    {train_df.isnull().sum().to_dict()}")

    # Check for duplicates
    train_dupes = train_df.duplicated(subset=["document"]).sum()
    test_dupes = test_df.duplicated(subset=["document"]).sum()
    print(f"\n  Duplicate reviews in training set: {train_dupes:,}")
    print(f"  Duplicate reviews in test set:     {test_dupes:,}")

    # --- Clean the data ---
    # Step 3a: Remove rows where the review text ('document' column) is missing
    train_df = train_df.dropna(subset=["document"])
    test_df = test_df.dropna(subset=["document"])

    # Step 3b: Remove duplicate reviews (keep the first occurrence)
    train_df = train_df.drop_duplicates(subset=["document"], keep="first")
    test_df = test_df.drop_duplicates(subset=["document"], keep="first")

    # Step 3c: Remove reviews that are empty strings (just whitespace)
    # str.strip() removes leading/trailing spaces, then we check length > 0
    train_df = train_df[train_df["document"].str.strip().str.len() > 0]
    test_df = test_df[test_df["document"].str.strip().str.len() > 0]

    # Reset the index so it goes 0, 1, 2, ... again after dropping rows
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --- Show results AFTER cleaning ---
    print("\n✅ After cleaning:")
    print(f"  Training set size: {len(train_df):,}")
    print(f"  Test set size:     {len(test_df):,}")

    # Show sample reviews from each class so we can see what they look like
    print("\n📝 Sample NEGATIVE reviews (label=0):")
    neg_samples = train_df[train_df["label"] == 0].head(3)
    for _, row in neg_samples.iterrows():
        print(f"  ❌ {row['document']}")

    print("\n📝 Sample POSITIVE reviews (label=1):")
    pos_samples = train_df[train_df["label"] == 1].head(3)
    for _, row in pos_samples.iterrows():
        print(f"  ✅ {row['document']}")

    return train_df, test_df


# ============================================================
# STEP 4 — PREPARE THE DATA FOR TRAINING (TOKENIZATION)
# ============================================================
# Machine learning models can't read text directly — they work with numbers.
# "Tokenization" converts text into numbers that the model can understand.
#
# Example: "이 영화 좋아요" → [2, 1378, 2495, 8834, 3]
#
# We use the tokenizer that matches our pretrained model (KoELECTRA).
# The tokenizer knows how to break Korean text into the right pieces.
def tokenize_data(train_df, test_df):
    """Tokenize the review text using the pretrained model's tokenizer."""
    print("\n" + "=" * 60)
    print("STEP 4 — TOKENIZING THE DATA")
    print("=" * 60)

    # Load the tokenizer that was trained alongside our pretrained model.
    # It knows Korean vocabulary and how to split Korean sentences.
    print(f"\n📦 Loading tokenizer for: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # If we're doing a quick test, take only a small subset of the data
    if not FULL_TRAINING:
        print(f"\n⚡ Quick mode: using {QUICK_SAMPLE_SIZE:,} training samples")
        train_df = train_df.sample(n=QUICK_SAMPLE_SIZE, random_state=42).reset_index(
            drop=True
        )
        # Use a proportional test set (20% of training size)
        test_sample_size = min(QUICK_SAMPLE_SIZE // 5, len(test_df))
        test_df = test_df.sample(n=test_sample_size, random_state=42).reset_index(
            drop=True
        )
        print(f"  Training samples: {len(train_df):,}")
        print(f"  Test samples:     {len(test_df):,}")

    # Convert pandas DataFrames back to HuggingFace Datasets.
    # The Trainer API works best with HuggingFace Dataset objects.
    train_dataset = Dataset.from_pandas(
        train_df[["document", "label"]].rename(columns={"document": "text"})
    )
    test_dataset = Dataset.from_pandas(
        test_df[["document", "label"]].rename(columns={"document": "text"})
    )

    # Define our tokenization function.
    # - padding="max_length": pad shorter reviews to the same length
    # - truncation=True: cut off reviews that are too long
    # - max_length=128: max number of tokens (most reviews are shorter than this)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    # Apply tokenization to all reviews at once (batched for speed)
    print("\n⏳ Tokenizing training data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    print("⏳ Tokenizing test data...")
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Tell HuggingFace which columns are the model inputs
    # The model needs: input_ids, attention_mask, and labels
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # Set the format to PyTorch tensors (the framework our model uses)
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    print(f"\n✅ Tokenization complete!")
    print(f"  Training dataset: {len(train_dataset):,} samples")
    print(f"  Test dataset:     {len(test_dataset):,} samples")

    # Show a tokenized example so we can see what happened
    example = train_dataset[0]
    print(f"\n📝 Example tokenized review:")
    print(f"  input_ids shape:      {example['input_ids'].shape}")
    print(f"  attention_mask shape:  {example['attention_mask'].shape}")
    print(f"  label:                {example['labels'].item()}")

    return tokenizer, train_dataset, test_dataset


# ============================================================
# STEP 5 — TRAIN THE SENTIMENT CLASSIFIER
# ============================================================
# "Fine-tuning" means taking a model that already understands Korean
# and teaching it specifically to classify movie reviews as positive/negative.
#
# Think of it like this:
#   - The pretrained model already knows Korean (like a Korean speaker)
#   - We're now teaching it to be a movie critic
#
# We use HuggingFace's Trainer API which handles the training loop for us.
def train_model(train_dataset, test_dataset):
    """Fine-tune the pretrained KoELECTRA model on our dataset."""
    print("\n" + "=" * 60)
    print("STEP 5 — TRAINING THE SENTIMENT CLASSIFIER")
    print("=" * 60)

    # Load the pretrained model for sequence classification.
    # num_labels=2 because we have two classes: negative (0) and positive (1)
    print(f"\n📦 Loading pretrained model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # Define a function to compute metrics during training.
    # This tells us how well the model is doing at each evaluation step.
    def compute_metrics(eval_pred):
        # eval_pred contains the model's predictions and the true labels
        logits, labels = eval_pred
        # Take the argmax to get the predicted class (0 or 1)
        predictions = np.argmax(logits, axis=-1)
        # Calculate various metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # TrainingArguments configures HOW the model is trained.
    # Think of these as the "settings" for training.
    training_args = TrainingArguments(
        # Where to save checkpoints during training
        output_dir="./results",
        # How many times to go through the entire dataset
        # More epochs = model sees the data more times = potentially better
        num_train_epochs=3,
        # How many samples to process at once
        # Larger batch = faster training but uses more GPU memory
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        # How quickly the model learns (too high = unstable, too low = too slow)
        learning_rate=2e-5,
        # Gradually reduce learning rate during training (helps convergence)
        warmup_steps=100,
        weight_decay=0.01,
        # Evaluate the model every 200 steps to track progress
        eval_strategy="steps",
        eval_steps=200,
        # Save a checkpoint every 200 steps
        save_strategy="steps",
        save_steps=200,
        # Only keep the best 2 checkpoints to save disk space
        save_total_limit=2,
        # Load the best model at the end (based on evaluation loss)
        load_best_model_at_end=True,
        # Log training metrics every 50 steps
        logging_steps=50,
        # Disable external reporting (we just want console output)
        report_to="none",
        # Use fp16 (half precision) if a GPU is available — speeds up training
        fp16=False,  # Set to True if you have a CUDA GPU
    )

    # The Trainer puts everything together and manages the training loop.
    # It handles: forward pass, loss calculation, backpropagation, optimization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Check if there's a checkpoint to resume from.
    # This way if you stop training in the middle, you can continue from where you left off!
    #
    # How it works: we look in the 'results/' folder for checkpoint folders.
    # Checkpoints are saved every 200 steps (like save points in a video game).
    # We pick the latest one and continue from there.
    checkpoint_dir = "./results"
    resume_from = None

    if os.path.isdir(checkpoint_dir):
        # Find all checkpoint-XXXX folders and sort by step number
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if checkpoints:
            # Sort by the step number (the number after "checkpoint-")
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_from = checkpoints[-1]  # Take the latest checkpoint
            print(f"\n🔁 Found checkpoint! Resuming from: {resume_from}")
        else:
            print("\n🆕 No checkpoint found. Starting training from scratch.")
    else:
        print("\n🆕 No checkpoint found. Starting training from scratch.")

    # Start training! 🚀
    print("\n🚀 Starting training...")
    print(f"  Epochs:     {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Train size: {len(train_dataset):,} samples")
    print(f"  Test size:  {len(test_dataset):,} samples")
    print()

    # Pass the checkpoint path to trainer.train() if we found one.
    # If resume_from is None, it starts fresh. If it's a path, it continues.
    trainer.train(resume_from_checkpoint=resume_from)

    print("\n✅ Training complete!")

    return model, trainer


# ============================================================
# STEP 6 — EVALUATE THE MODEL
# ============================================================
# Now let's see how well our model actually performs!
# We'll look at:
#   - Accuracy: % of reviews correctly classified
#   - Precision: of all reviews predicted as positive, how many really are?
#   - Recall: of all truly positive reviews, how many did we catch?
#   - F1 Score: harmonic mean of precision and recall (balanced metric)
#   - Confusion Matrix: shows exactly what the model got right and wrong
def evaluate_model(trainer, test_dataset):
    """Evaluate the trained model and print detailed metrics."""
    print("\n" + "=" * 60)
    print("STEP 6 — EVALUATING THE MODEL")
    print("=" * 60)

    # Run the model on the test set and get predictions
    predictions = trainer.predict(test_dataset)

    # Extract predictions and true labels
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Calculate all metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    # Print results in a nice format
    print("\n📊 Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"  Precision: {precision:.4f}  ({precision * 100:.1f}%)")
    print(f"  Recall:    {recall:.4f}  ({recall * 100:.1f}%)")
    print(f"  F1 Score:  {f1:.4f}  ({f1 * 100:.1f}%)")

    # Confusion Matrix — a 2x2 grid showing:
    #   [True Negatives,  False Positives]
    #   [False Negatives, True Positives ]
    cm = confusion_matrix(labels, preds)
    print("\n📊 Confusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"  Actually Neg:       {cm[0][0]:>6,}              {cm[0][1]:>6,}")
    print(f"  Actually Pos:       {cm[1][0]:>6,}              {cm[1][1]:>6,}")

    # Detailed classification report from scikit-learn
    print("\n📊 Detailed Classification Report:")
    target_names = ["Negative (0)", "Positive (1)"]
    print(classification_report(labels, preds, target_names=target_names))

    return accuracy, precision, recall, f1


# ============================================================
# STEP 7 — TEST WITH CUSTOM INPUT
# ============================================================
# This function lets you type any Korean movie review and the model
# will tell you if it's positive or negative, with a confidence score.
def predict_sentiment(text, model, tokenizer):
    """
    Predict whether a Korean review is positive or negative.

    Args:
        text (str): A Korean movie review (e.g. "이 영화 정말 재미있어요!")
        model: The trained sentiment classification model
        tokenizer: The tokenizer matching the model

    Returns:
        dict: {
            'text': the input text,
            'sentiment': 'Positive' or 'Negative',
            'confidence': a float between 0 and 1,
            'label': 0 (negative) or 1 (positive)
        }
    """
    import torch

    # Move model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Tokenize the input text (same way we tokenized the training data)
    inputs = tokenizer(
        text,
        return_tensors="pt",  # Return PyTorch tensors
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    # Move inputs to the same device as the model (CPU or GPU)
    device = next(model.parameters()).device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Run the model (no_grad = we don't need gradients for prediction)
    with torch.no_grad():
        outputs = model(**inputs)

    # The model outputs "logits" — raw scores for each class.
    # We use softmax to convert them to probabilities (0 to 1, summing to 1).
    probabilities = torch.softmax(outputs.logits, dim=-1)

    # Get the predicted class (0 or 1) and its confidence
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_label].item()

    # Map the numeric label to a human-readable sentiment
    sentiment = "Positive ✅" if predicted_label == 1 else "Negative ❌"

    result = {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "label": predicted_label,
    }

    return result


# ============================================================
# STEP 8 — SAVE THE MODEL
# ============================================================
# Save the fine-tuned model and tokenizer so we can reuse them
# without having to retrain every time.
def save_model(model, tokenizer):
    """Save the trained model and tokenizer to the models/ folder."""
    print("\n" + "=" * 60)
    print("STEP 8 — SAVING THE MODEL")
    print("=" * 60)

    # Create the save directory if it doesn't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Save the model weights and configuration
    model.save_pretrained(MODEL_SAVE_DIR)

    # Save the tokenizer (vocabulary and settings)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    print(f"\n✅ Model saved to: {MODEL_SAVE_DIR}")
    print("  You can load it later with:")
    print(f'    model = AutoModelForSequenceClassification.from_pretrained("{MODEL_SAVE_DIR}")')
    print(f'    tokenizer = AutoTokenizer.from_pretrained("{MODEL_SAVE_DIR}")')


# ============================================================
# MAIN — RUN THE FULL PIPELINE
# ============================================================
if __name__ == "__main__":
    print("🎬 Korean Movie Review Sentiment Analyzer")
    print("=" * 60)

    if FULL_TRAINING:
        print("📢 Mode: FULL TRAINING (using all ~150k samples)")
    else:
        print(f"📢 Mode: QUICK TEST (using {QUICK_SAMPLE_SIZE:,} samples)")
    print()

    # Step 2: Load the dataset
    train_df, test_df = load_data()

    # Step 3: Clean the data
    train_df, test_df = clean_data(train_df, test_df)

    # Step 4: Tokenize the data
    tokenizer, train_dataset, test_dataset = tokenize_data(train_df, test_df)

    # Step 5: Train the model
    model, trainer = train_model(train_dataset, test_dataset)

    # Step 6: Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(trainer, test_dataset)

    # Step 7: Test with custom Korean movie reviews
    print("\n" + "=" * 60)
    print("STEP 7 — TESTING WITH CUSTOM REVIEWS")
    print("=" * 60)

    # Here are some sample reviews to test with.
    # You can add your own Korean reviews here!
    test_reviews = [
        "이 영화 정말 재미있어요! 배우들 연기도 최고!",       # This movie is really fun! Acting is the best!
        "완전 별로... 시간 낭비했다.",                          # Totally bad... waste of time.
        "그저 그런 영화. 나쁘지도 좋지도 않다.",               # So-so movie. Neither bad nor good.
        "역대 최고의 한국 영화! 꼭 보세요!",                   # Best Korean movie ever! Must watch!
        "스토리가 너무 지루하고 연기가 어색해요.",             # The story is boring and the acting is awkward.
    ]

    print("\n🎯 Testing with sample Korean reviews:\n")
    for review in test_reviews:
        result = predict_sentiment(review, model, tokenizer)
        print(f"  Review:     {result['text']}")
        print(f"  Sentiment:  {result['sentiment']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print()

    # Step 8: Save the model
    save_model(model, tokenizer)

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)
    print(f"\n  Model accuracy:  {accuracy:.1%}")
    print(f"  Model saved to:  {MODEL_SAVE_DIR}")
    print(f"\n  To use the model later, run:")
    print(f"    from transformers import AutoTokenizer, AutoModelForSequenceClassification")
    print(f'    tokenizer = AutoTokenizer.from_pretrained("{MODEL_SAVE_DIR}")')
    print(f'    model = AutoModelForSequenceClassification.from_pretrained("{MODEL_SAVE_DIR}")')
    print()