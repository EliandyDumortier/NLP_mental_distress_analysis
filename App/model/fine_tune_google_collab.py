import os
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === COLAB: Set base directory ===
BASE_DIR = "/content"
DATA_PATH = os.path.join(BASE_DIR, "emotional_distress_dataset_combined.csv")

## === Custom Callback to Track Training History ===
class TrainingHistoryCallback(TrainerCallback):
    def __init__(self):
        self.training_history = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {'epoch': state.epoch, 'step': state.global_step}
            entry.update(logs)
            self.training_history.append(entry)

## === STEP 1: Load, Clean & Balance ===
print("🔄 Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["text", "distress_level"], inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Map labels
label2id = {"no_distress": 0, "mild": 1, "moderate": 2, "severe": 3}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["distress_level"].map(label2id).astype(int)

# Upsample minority classes to match majority
counts = df["label"].value_counts()
max_count = counts.max()
df = pd.concat([
    resample(df[df["label"] == lbl],
             replace=True,
             n_samples=max_count,
             random_state=42)
    for lbl in counts.index
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Balanced dataset size: {len(df)}")
print(f"📊 New label distribution:\n{df['label'].value_counts()}")

# Train/Val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(),
    test_size=0.2, stratify=df["label"], random_state=42
)
print(f"📈 Training samples: {len(train_texts)}")
print(f"📊 Validation samples: {len(val_texts)}")

## === STEP 2: Tokenization ===
print("🔄 Tokenizing...")
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=128
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=128
)

## === STEP 3: Dataset Wrapper ===
class DistressDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = DistressDataset(train_encodings, train_labels)
val_dataset   = DistressDataset(val_encodings, val_labels)

## === STEP 4: Model Configuration ===
print("🔄 Configuring model...")
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, config=config
)
model.gradient_checkpointing_enable()

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "f1":        f1_score(pred.label_ids, preds, average="weighted"),
        "precision": precision_score(pred.label_ids, preds, average="weighted"),
        "recall":    recall_score(pred.label_ids, preds, average="weighted")
    }

## === STEP 5: Training Configuration ===
print("🔄 Setting up training arguments and class weights...")
checkpoints_dir = os.path.join(BASE_DIR, "checkpoints")
results_dir     = os.path.join(BASE_DIR, "results")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label"]),
    y=df["label"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    label_smoothing_factor=0.1,
)

history_callback = TrainingHistoryCallback()

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(model.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
               history_callback]
)

## === STEP 6: Training ===
print("🚀 Starting training...")
trainer.train()
print("✅ Training completed!")

## === STEP 7: Final Evaluation ===
print("🔄 Evaluating on validation set...")
val_preds = trainer.predict(val_dataset)
final_preds = np.argmax(val_preds.predictions, axis=1)

final_accuracy  = accuracy_score(val_labels, final_preds)
final_f1        = f1_score(val_labels, final_preds, average="weighted")
final_precision = precision_score(val_labels, final_preds, average="weighted")
final_recall    = recall_score(val_labels, final_preds, average="weighted")

print(f"📊 Final Results – Acc: {final_accuracy:.4f}, F1: {final_f1:.4f}, "
      f"Prec: {final_precision:.4f}, Rec: {final_recall:.4f}")
print(classification_report(val_labels, final_preds, target_names=list(id2label.values())))

## === STEP 8: Save Model & JSON Results ===
print("💾 Saving model and results...")
model_save_path = os.path.join(BASE_DIR, "distress_classifier")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

training_results = {
    "model_name": model_name,
    "training_samples": len(train_texts),
    "validation_samples": len(val_texts),
    "final_accuracy":  final_accuracy,
    "final_f1":        final_f1,
    "final_precision": final_precision,
    "final_recall":    final_recall,
    "training_history": history_callback.training_history,
    "classification_report":
        classification_report(val_labels, final_preds,
                              target_names=list(id2label.values()),
                              output_dict=True),
    "confusion_matrix": confusion_matrix(val_labels, final_preds).tolist(),
    "label_mapping":    {"label2id": label2id, "id2label": id2label},
    "training_config": {
        "learning_rate":    training_args.learning_rate,
        "batch_size":       training_args.per_device_train_batch_size,
        "epochs":           training_args.num_train_epochs,
        "weight_decay":     training_args.weight_decay
    }
}
with open(os.path.join(BASE_DIR, "training_results.json"), "w") as f:
    json.dump(training_results, f, indent=2)
print("✅ Results saved to training_results.json")

## === STEP 9: Generate Visualizations ===
print("🎨 Generating visualizations...")

# Training history (Plotly interactive)
if history_callback.training_history:
    hist_df = pd.DataFrame(history_callback.training_history)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Train Loss", "Val Accuracy", "Val F1", "Learning Rate")
    )
    if "train_loss" in hist_df:
        fig.add_trace(go.Scatter(x=hist_df.epoch, y=hist_df.train_loss,
                                 name="Train Loss"), row=1, col=1)
    if "eval_accuracy" in hist_df:
        fig.add_trace(go.Scatter(x=hist_df.epoch, y=hist_df.eval_accuracy,
                                 name="Val Acc"), row=1, col=2)
    if "eval_f1" in hist_df:
        fig.add_trace(go.Scatter(x=hist_df.epoch, y=hist_df.eval_f1,
                                 name="Val F1"), row=2, col=1)
    if "learning_rate" in hist_df:
        fig.add_trace(go.Scatter(x=hist_df.epoch, y=hist_df.learning_rate,
                                 name="LR"), row=2, col=2)
    fig.update_layout(title_text="Training History", showlegend=False)
    fig.write_html(os.path.join(results_dir, "training_history.html"))

# Confusion matrix
cm = confusion_matrix(val_labels, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(id2label.values()),
            yticklabels=list(id2label.values()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# Metrics summary bar
metrics_data = {
    "Metric": ["Accuracy", "F1-Score", "Precision", "Recall"],
    "Score":  [final_accuracy, final_f1, final_precision, final_recall]
}
fig2 = px.bar(metrics_data, x="Metric", y="Score",
              title="Model Performance Metrics",
              color="Score", color_continuous_scale="Viridis")
fig2.update_layout(yaxis=dict(range=[0.8, 1.0]))
fig2.write_html(os.path.join(results_dir, "metrics_summary.html"))

print("🎉 All visuals saved in results/")  
