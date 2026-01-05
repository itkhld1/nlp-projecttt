import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np
import os

# turning off this warning thing because it was kinda annoying
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# making a dictionary for the categories so the model knows what each number means
id2label = {
    0: 'Shopping',
    1: 'Dining Out',
    2: 'Entertainment',
    3: 'Transportation',
    4: 'Housing',
    5: 'Payments/Credits',
    6: 'Utilities',
    7: 'Service Subscriptions'
}

# doing the reverse one too because trainer needs it
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

print("step 1 done, labels are ready")

print("loading dataset from huggingface...")
dataset = load_dataset("rajeshradhakrishnan/fin-transaction-category", split='train')

# changing column names so trainer doesn’t complain
dataset = dataset.rename_column("merchant", "text")
dataset = dataset.rename_column("category", "label")

# splitting the dataset into train and test
dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset = dataset_splits['train']
eval_dataset = dataset_splits['test']

print(f"dataset loaded. train = {len(train_dataset)}, test = {len(eval_dataset)}")

# loading the tokenizer for finbert
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# this tokenizes the text so the model can actually read it
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("tokenizing stuff now...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

print("loading the finbert model (fingers crossed)")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# this calculates accuracy after predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# training settings, kinda default values
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# trainer thing that actually does the training work
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("starting training... might take a while")
trainer.train()

# saving the final model so we can use it later
final_model_path = "./my_finbert_classifier"
trainer.save_model(final_model_path)
print(f"done!! model saved here -> {final_model_path}")