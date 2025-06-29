import os
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
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
from sklearn.utils.class_weight import compute_class_weight

# Prevent system sleep (Linux, macOS, Windows)
import platform
import subprocess
import atexit

if platform.system() == "Linux":
    # Start a background process to inhibit sleep
    inhibitor = subprocess.Popen(["caffeinate", "-d"])
    atexit.register(inhibitor.terminate)
elif platform.system() == "Darwin":
    inhibitor = subprocess.Popen(["caffeinate"])
    atexit.register(inhibitor.terminate)
elif platform.system() == "Windows":
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

## === Custom Callback to Track Training History ===
class TrainingHistoryCallback(TrainerCallback):
    def __init__(self):
        self.training_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Collect training metrics
            history_entry = {
                'epoch': state.epoch,
                'step': state.global_step,
            }
            
            # Add available metrics
            for key, value in logs.items():
                history_entry[key] = value
            
            self.training_history.append(history_entry)

## === STEP 1: Load & Preprocess ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "App/data/emotional_distress_dataset_combined.csv")

print("🔄 Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "distress_level"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

label2id = {"no_distress": 0, "mild": 1, "moderate": 2, "severe": 3}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["distress_level"].map(label2id)

# Drop rows where mapping failed (label is NaN)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print(f"✅ Dataset loaded with {len(df)} samples.")
print(f"📊 Label distribution:\n{df['distress_level'].value_counts()}")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), 
    test_size=0.2, stratify=df["label"], random_state=42
)

print(f"📈 Training samples: {len(train_texts)}")
print(f"📊 Validation samples: {len(val_texts)}")

## === STEP 2: Tokenization ===
print("🔄 Loading tokenizer and preparing data...")
#model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# For faster training, reduce max_length (if your texts are not very long)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

## === STEP 3: Dataset Wrapper ===
class DistressDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = DistressDataset(train_encodings, train_labels)
val_dataset = DistressDataset(val_encodings, val_labels)

## === STEP 4: Model Configuration ===
print("🔄 Configuring model...")
config = AutoConfig.from_pretrained(
    model_name, 
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
    attention_probs_dropout_prob=0.4,
    hidden_dropout_prob=0.4
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    config=config, 
    ignore_mismatched_sizes=True
)

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(pred.label_ids, predictions)
    f1 = f1_score(pred.label_ids, predictions, average="weighted")
    precision = precision_score(pred.label_ids, predictions, average="weighted")
    recall = recall_score(pred.label_ids, predictions, average="weighted")
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

## === STEP 5: Training Configuration ===
print("🔄 Setting up training configuration...")

# Create directories
checkpoints_dir = os.path.join(BASE_DIR, "App/model/checkpoints")
results_dir = os.path.join(BASE_DIR, "App/model/results")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float)

training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    evaluation_strategy="epoch",      # more frequent eval
    eval_steps=1000,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.2,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    label_smoothing_factor=0.15,
)

# Initialize callbacks
history_callback = TrainingHistoryCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2),  # <--- stricter early stopping
        history_callback
    ]
)

## === STEP 6: Training ===
print("🚀 Starting training...")
print("=" * 50)

training_output = trainer.train()

print("✅ Training completed!")
print("=" * 50)

## === STEP 7: Final Evaluation ===
print("🔄 Performing final evaluation...")

# Get final predictions
val_predictions = trainer.predict(val_dataset)
final_preds = np.argmax(val_predictions.predictions, axis=1)

# Calculate final metrics
final_accuracy = accuracy_score(val_labels, final_preds)
final_f1 = f1_score(val_labels, final_preds, average="weighted")
final_precision = precision_score(val_labels, final_preds, average="weighted")
final_recall = recall_score(val_labels, final_preds, average="weighted")

print(f"📊 Final Results:")
print(f"   Accuracy:  {final_accuracy:.4f}")
print(f"   F1-Score:  {final_f1:.4f}")
print(f"   Precision: {final_precision:.4f}")
print(f"   Recall:    {final_recall:.4f}")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(val_labels, final_preds, target_names=list(id2label.values())))

## === STEP 8: Save Model and Results ===
print("💾 Saving model and results...")

# Save the model
model_save_path = os.path.join(BASE_DIR, "App/model/distress_classifier")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Prepare training results
training_results = {
    "model_name": model_name,
    "training_samples": len(train_texts),
    "validation_samples": len(val_texts),
    "final_accuracy": final_accuracy,
    "final_f1": final_f1,
    "final_precision": final_precision,
    "final_recall": final_recall,
    "training_history": history_callback.training_history,
    "classification_report": classification_report(val_labels, final_preds, target_names=list(id2label.values()), output_dict=True),
    "confusion_matrix": confusion_matrix(val_labels, final_preds).tolist(),
    "label_mapping": {
        "label2id": label2id,
        "id2label": id2label
    },
    "training_config": {
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "epochs": training_args.num_train_epochs,
        "weight_decay": training_args.weight_decay
    }
}

# Save training results
results_file = os.path.join(BASE_DIR, "App/model/training_results.json")
with open(results_file, 'w') as f:
    json.dump(training_results, f, indent=2)

print(f"✅ Training results saved to: {results_file}")

## === STEP 9: Generate Visualizations ===
print("🎨 Generating visualizations...")

# 1. Training History Plot
if history_callback.training_history:
    history_df = pd.DataFrame(history_callback.training_history)
    
    # Training loss over time
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Accuracy', 'F1 Score', 'Learning Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if 'train_loss' in history_df.columns:
        fig.add_trace(
            go.Scatter(x=history_df['epoch'], y=history_df['train_loss'], 
                      name='Training Loss', line=dict(color='red')),
            row=1, col=1
        )
    
    if 'eval_accuracy' in history_df.columns:
        fig.add_trace(
            go.Scatter(x=history_df['epoch'], y=history_df['eval_accuracy'], 
                      name='Validation Accuracy', line=dict(color='blue')),
            row=1, col=2
        )
    
    if 'eval_f1' in history_df.columns:
        fig.add_trace(
            go.Scatter(x=history_df['epoch'], y=history_df['eval_f1'], 
                      name='F1 Score', line=dict(color='green')),
            row=2, col=1
        )
    
    if 'learning_rate' in history_df.columns:
        fig.add_trace(
            go.Scatter(x=history_df['epoch'], y=history_df['learning_rate'], 
                      name='Learning Rate', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Training History",
        showlegend=False,
        height=600
    )
    
    # Save plot
    training_plot_path = os.path.join(results_dir, "training_history.html")
    fig.write_html(training_plot_path)
    print(f"📊 Training history plot saved to: {training_plot_path}")

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(val_labels, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(id2label.values()),
            yticklabels=list(id2label.values()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 Confusion matrix saved to: {confusion_matrix_path}")

# 3. Metrics Summary
metrics_data = {
    'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
    'Score': [final_accuracy, final_f1, final_precision, final_recall]
}

fig = px.bar(metrics_data, x='Metric', y='Score', 
             title='Model Performance Metrics',
             color='Score',
             color_continuous_scale='Viridis')
fig.update_layout(yaxis=dict(range=[0.8, 1.0]))

metrics_plot_path = os.path.join(results_dir, "metrics_summary.html")
fig.write_html(metrics_plot_path)
print(f"📊 Metrics summary saved to: {metrics_plot_path}")

print("🎉 Training completed successfully!")
print(f"📁 Model saved to: {model_save_path}")
print(f"📁 Results saved to: {results_dir}")
print("=" * 50)

