from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class GoEmotionsClassifier:
    def __init__(self):
        self.model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
            'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

    def predict(self, text: str, top_k: int = 3):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        top_probs, top_labels = torch.topk(probs, top_k)

        return [
            {"label": self.labels[i], "score": float(prob)}
            for i, prob in zip(top_labels[0], top_probs[0])
        ]
