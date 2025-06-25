import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score

## === STEP 1: Load & Preprocess ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "App/data/emotional_distress_dataset_combined.csv")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "distress_level"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

label2id = {"no_distress": 0, "mild": 1, "moderate": 2, "severe": 3}
df["label"] = df["distress_level"].map(label2id)

# Drop rows where mapping failed (label is NaN)
df = df.dropna(subset=["label"])
print(f"✅ Dataset loaded with {len(df)} samples.")
df["label"] = df["label"].astype(int)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, stratify=df["label"], random_state=42
)

## === STEP 2: Tokenization ===
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

## === STEP 3: Dataset Wrapper ===
class DistressDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {"labels": torch.tensor(self.labels[idx])}

train_dataset = DistressDataset(train_encodings, train_labels)
val_dataset = DistressDataset(val_encodings, val_labels)

## === STEP 4: Model & Trainer ===
config = AutoConfig.from_pretrained(model_name, num_labels=4, attention_probs_dropout_prob=0.3)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, config=config, ignore_mismatched_sizes=True
)

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "f1": f1_score(pred.label_ids, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "App/model/checkpoints"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=False,# Optional: if GPU supports
    #label_smoothing=0.1  
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

trainer.save_model(os.path.join(BASE_DIR, "App/model/distress_classifier"))
tokenizer.save_pretrained(os.path.join(BASE_DIR, "App/model/distress_classifier"))


########################## First version of the script ##########################

# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# """This script fine-tunes a DistilBERT model on a custom emotional distress dataset.
# It loads the dataset, tokenizes the text, and prepares it for training with PyTorch.
# """
# ## STEP 1: Load and Preprocess Dataset
# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# DATA_PATH = os.path.join(BASE_DIR, "App/data/emotional_distress_dataset_combined.csv")

# # Load dataset
# df = pd.read_csv(DATA_PATH)

# # Map labels to integers
# label2id = {
#     "no_distress": 0,
#     "mild": 1,
#     "moderate": 2,
#     "severe": 3
# }
# # Apply it
# df["label"] = df["distress_level"].map(label2id)
# df = df.dropna(subset=["label"])
# df["label"] = df["label"].astype(int)

# # Split into train/test
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     df["text"].tolist(), df["label"].tolist(), test_size=0.1, stratify=df["label"], random_state=42
# )

# ## # STEP 2: Tokenization
# # Ensure transformers library is installed
# # You can uncomment the next line to install it if running this script directly
# # !pip install transformers
# from transformers import AutoTokenizer

# # Load tokenizer (you can change the model name if needed)
# model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# # Tokenize datasets
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# ## #STEP 3: Create Dataset objects
# import torch
# from torch.utils.data import Dataset

# class DistressDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, idx):
#         return {
#             key: torch.tensor(val[idx]) for key, val in self.encodings.items()
#         } | {"labels": torch.tensor(self.labels[idx])}

# # Wrap into Hugging Face-compatible dataset
# train_dataset = DistressDataset(train_encodings, train_labels)
# val_dataset = DistressDataset(val_encodings, val_labels)

# ## STEP 4: Load Model + Set Up Trainer

# from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score

# # Load config first and override the num_labels
# config = AutoConfig.from_pretrained(model_name, num_labels=4)

# # Load model with correct num_labels
# model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir=os.path.join(BASE_DIR, "App/model/checkpoints"),
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_strategy="epoch",
#     save_total_limit=2,
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=32,
#     num_train_epochs=4,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1"
# )

# # Metrics
# def compute_metrics(pred):
#     preds = np.argmax(pred.predictions, axis=1)
#     labels = pred.label_ids
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds, average="weighted")
#     }

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

# ## # STEP 5: Train the model
# trainer.train()

# # Optional: Save model and tokenizer
# trainer.save_model(os.path.join(BASE_DIR, "App/model/distress_classifier"))
# tokenizer.save_pretrained(os.path.join(BASE_DIR, "App/model/distress_classifier"))
