# 🎬 Korean Movie Review Sentiment Analyzer

A machine learning project that fine-tunes a pretrained Korean language model ([KoELECTRA](https://huggingface.co/monologg/koelectra-base-finetuned-sentiment)) on the **NSMC dataset** (200,000 Korean movie reviews) to classify reviews as **positive** or **negative**.

> Built as a portfolio project to demonstrate practical NLP and transfer learning skills.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Key Concepts Explained](#key-concepts-explained)
- [What I Learned](#what-i-learned)

---

## Overview

**Goal:** Given a Korean movie review (plain text), predict whether it expresses a **positive** (👍) or **negative** (👎) sentiment.

**Approach:** Instead of training a model from scratch (which requires massive data and compute), I used **transfer learning** — taking a model that already understands Korean and teaching it to classify sentiment.

**Example:**
```
Input:  "이 영화 정말 재미있어요! 배우들 연기도 최고!"
Output: Positive ✅ (confidence: 92.3%)

Input:  "완전 별로... 시간 낭비했다."
Output: Negative ❌ (confidence: 88.7%)
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│                                                             │
│  Korean Review Text                                         │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐    Converts text to numbers                   │
│  │Tokenizer │    "좋아요" → [2, 1378, 8834, 3]              │
│  └──────────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────┐    Pretrained Korean language model        │
│  │  KoELECTRA   │    Already understands Korean grammar,    │
│  │  (ELECTRA)   │    vocabulary, and context                │
│  └──────────────┘                                           │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────┐    Added on top of KoELECTRA              │
│  │ Classifier   │    Learns to map language understanding   │
│  │   Head       │    → positive/negative                    │
│  └──────────────┘                                           │
│       │                                                     │
│       ▼                                                     │
│  Prediction: [0.15, 0.85] → Positive (85% confidence)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

**NSMC (Naver Sentiment Movie Corpus)** — one of the most widely used Korean NLP benchmark datasets.

| Split    | Samples  | Description                          |
|----------|----------|--------------------------------------|
| Training | 150,000  | Reviews used to train the model     |
| Test     | 50,000   | Reviews used to evaluate the model  |

- **Source:** [github.com/e9t/nsmc](https://github.com/e9t/nsmc)
- **Labels:** Binary — `0` (negative) and `1` (positive)
- **Balance:** Nearly 50/50 split (well-balanced)

### Data Cleaning
Before training, the data is cleaned:
- ❌ **5 missing values** removed (empty review text)
- ❌ **3,817 duplicate reviews** removed from training set
- ❌ **Empty strings** (whitespace-only reviews) removed
- ✅ Final training set: **146,182 reviews**

---

## Model Architecture

### Why KoELECTRA?

| Model | Pros | Cons |
|-------|------|------|
| Train from scratch | Full control | Needs millions of samples, weeks of training |
| **KoELECTRA (chosen)** | **Already understands Korean, fast to fine-tune** | Requires GPU for full training |
| KR-FinBert-SC | Good at Korean | Designed for financial text, not general sentiment |

**KoELECTRA** is an [ELECTRA](https://arxiv.org/abs/2003.10555)-based model pretrained on a large Korean text corpus. ELECTRA models are trained using a "replaced token detection" approach, which is more sample-efficient than BERT's masked language modeling.

### Fine-tuning Process
```
Pretrained KoELECTRA (knows Korean)
        +
Classification Head (2 outputs: negative, positive)
        ↓
Train on NSMC for up to 10 epochs (early stopping on eval_loss)
        ↓
Fine-tuned Sentiment Classifier (90.2% accuracy)
```

### Training Hyperparameters
| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 2e-5 | Standard for fine-tuning transformers |
| Batch size | 32 | Balance between speed and memory |
| Max epochs | 10 (early stopping) | Stops automatically when eval_loss stops improving |
| Early stopping patience | 2 | Stop if no improvement for 2 evaluations in a row |
| Max token length | 128 | Most Korean reviews are shorter than this |
| Warmup steps | 100 | Prevents unstable early training |
| Weight decay | 0.01 | Regularization to prevent overfitting |

---

## Results

*Trained on the full dataset: **146,182 Korean movie reviews** (after cleaning).*

| Metric    | Score  |
|-----------|--------|
| Accuracy  | **90.2%** |
| Precision | **90.2%** |
| Recall    | **90.3%** |
| F1 Score  | **90.3%** |

This result is competitive with published results on the NSMC benchmark (typical range: 88–92%).

### Sample Predictions
| Review (Korean) | Translation | Predicted | Confidence |
|-----------------|-------------|-----------|------------|
| 이 영화 정말 재미있어요! 배우들 연기도 최고! | This movie is really fun! The acting is the best! | ✅ Positive | 99.4% |
| 완전 별로... 시간 낭비했다. | Totally bad... waste of time. | ❌ Negative | 99.5% |
| 역대 최고의 한국 영화! 꼭 보세요! | Best Korean movie ever! Must watch! | ✅ Positive | 99.3% |
| 스토리가 너무 지루하고 연기가 어색해요. | The story is boring and the acting is awkward. | ❌ Negative | 99.5% |

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- ~2GB disk space (for model weights and data)

### Setup
```bash
# Clone the repository
git clone https://github.com/cringepnh/korean-sentiment-analyzer.git
cd korean-sentiment-analyzer

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline
```bash
python main.py
```

This will:
1. Load the NSMC dataset (200k Korean movie reviews)
2. Clean and preprocess the data
3. Tokenize reviews using KoELECTRA's tokenizer
4. Fine-tune the model on the full 146k training set
5. Evaluate and print metrics (accuracy, F1, confusion matrix)
6. Test on sample Korean reviews
7. Save the trained model to `models/`

> 💡 To do a quick test run first, set `FULL_TRAINING = False` in `main.py` — trains on 5,000 samples in ~5 minutes.

> 💡 Training supports **checkpoint resume** — if you stop and restart, it continues from the last saved checkpoint automatically.

### Use as Standalone Predictor
After training, you can use the model directly in Python:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from main import predict_sentiment

# Load the saved model
model = AutoModelForSequenceClassification.from_pretrained("models/sentiment-model")
tokenizer = AutoTokenizer.from_pretrained("models/sentiment-model")

# Predict sentiment for any Korean text
result = predict_sentiment("이 영화 정말 좋아요!", model, tokenizer)
print(result)
# {'text': '이 영화 정말 좋아요!', 'sentiment': 'Positive ✅', 'confidence': 0.94, 'label': 1}
```

---

## Project Structure

```
korean-sentiment-analyzer/
├── data/
│   ├── ratings_train.txt       # 150k training reviews (TSV)
│   └── ratings_test.txt        # 50k test reviews (TSV)
├── models/
│   └── sentiment-model/        # Saved trained model (generated after training)
├── notebooks/                  # Jupyter notebooks (for experimentation)
├── main.py                     # Complete ML pipeline (all 8 steps)
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

### Key Files

| File | Description |
|------|-------------|
| `main.py` | The complete pipeline — from data loading to model saving |
| `requirements.txt` | All required Python packages |
| `data/ratings_train.txt` | 150,000 labeled Korean movie reviews for training |
| `data/ratings_test.txt` | 50,000 labeled Korean movie reviews for evaluation |

---

## Key Concepts Explained

### What is Transfer Learning?
Instead of training a model from scratch on our small dataset, we take a model that was already trained on millions of Korean sentences (KoELECTRA) and **fine-tune** it for our specific task. This is like hiring a Korean language expert and teaching them to do movie review analysis — much faster than teaching someone Korean from scratch.

### What is Tokenization?
Neural networks work with numbers, not text. A **tokenizer** converts Korean text into sequences of numbers using a learned vocabulary:
```
"이 영화 좋아요" → [2, 1378, 2495, 8834, 3]
```
Each number corresponds to a piece of a word (called a "subword token") in the model's vocabulary.

### What is Fine-tuning?
Fine-tuning adjusts the pretrained model's weights slightly so it becomes good at our specific task (sentiment classification). We use a small learning rate (2e-5) to make tiny adjustments without "forgetting" the Korean language knowledge.

### Evaluation Metrics
- **Accuracy**: Percentage of correct predictions overall
- **Precision**: Of all reviews predicted positive, how many actually are?
- **Recall**: Of all actually positive reviews, how many did the model find?
- **F1 Score**: Harmonic mean of precision and recall — a balanced single metric
- **Confusion Matrix**: A 2×2 grid showing exact counts of correct/incorrect predictions

---

## What I Learned

Building this project taught me:

1. **NLP Pipeline Design** — How to structure an end-to-end machine learning project: data loading → cleaning → preprocessing → training → evaluation → deployment.

2. **Transfer Learning** — Why fine-tuning pretrained models (like KoELECTRA) is more practical than training from scratch, especially with limited compute resources.

3. **Data Cleaning** — Real-world data is messy. Handling missing values, duplicates, and edge cases is a critical (and often underestimated) step.

4. **Tokenization** — How transformer models convert text into numerical representations, and why subword tokenization works well for Korean.

5. **HuggingFace Ecosystem** — Practical experience with the `transformers` and `datasets` libraries, which are industry-standard tools for NLP.

6. **Model Evaluation** — Understanding that accuracy alone isn't enough — precision, recall, and F1 give a fuller picture of model performance.

7. **Training Best Practices** — Learning rate warmup, weight decay, checkpoint saving, and how to resume interrupted training.

---

## Technologies Used

- **Python 3** — Primary programming language
- **PyTorch** — Deep learning framework
- **HuggingFace Transformers** — Pre-trained model library
- **KoELECTRA** — Korean ELECTRA language model
- **Pandas & NumPy** — Data manipulation
- **scikit-learn** — Evaluation metrics
- **NSMC Dataset** — Korean sentiment benchmark

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ as a machine learning portfolio project*
