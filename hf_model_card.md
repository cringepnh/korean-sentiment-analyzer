---
language:
- ko
license: mit
tags:
- korean
- sentiment-analysis
- text-classification
- koelectra
- nsmc
- fine-tuned
datasets:
- nsmc
metrics:
- accuracy
- f1
base_model: monologg/koelectra-base-finetuned-sentiment
---

# 🎬 KoELECTRA Korean Sentiment Analyzer

Fine-tuned Korean sentiment classification model based on [KoELECTRA](https://huggingface.co/monologg/koelectra-base-finetuned-sentiment), trained on the [NSMC dataset](https://github.com/e9t/nsmc) (Naver Sentiment Movie Corpus).

Classifies Korean movie reviews as **positive** (1) or **negative** (0).

**GitHub:** [cringepnh/korean-sentiment-analyzer](https://github.com/cringepnh/korean-sentiment-analyzer)

## 📊 Results

Evaluated on the full NSMC test set (49,157 samples after cleaning):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | **90.2%** |
| Precision | **90.2%** |
| Recall    | **90.3%** |
| F1 Score  | **90.3%** |

## 🚀 How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "cringepnh/koelectra-korean-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    confidence = probs[0][label].item()
    sentiment = "Positive ✅" if label == 1 else "Negative ❌"
    return {"sentiment": sentiment, "confidence": f"{confidence:.1%}"}

# Examples
print(predict("이 영화 정말 재미있어요! 배우들 연기도 최고!"))
# {'sentiment': 'Positive ✅', 'confidence': '99.4%'}

print(predict("완전 별로... 시간 낭비했다."))
# {'sentiment': 'Negative ❌', 'confidence': '99.5%'}
```

## 🏋️ Training Details

- **Base model:** `monologg/koelectra-base-finetuned-sentiment`
- **Dataset:** NSMC (146,182 training samples after cleaning)
- **Epochs:** Up to 10 with early stopping (patience=2 on eval_loss)
- **Batch size:** 32
- **Learning rate:** 2e-5
- **Max token length:** 128
- **Hardware:** CPU (laptop)

## 📁 Dataset

**NSMC (Naver Sentiment Movie Corpus)**
- 150,000 training / 50,000 test Korean movie reviews
- Binary labels: 0 (negative), 1 (positive)
- Source: [github.com/e9t/nsmc](https://github.com/e9t/nsmc)

## 📄 License

MIT
