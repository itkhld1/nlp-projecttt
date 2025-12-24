# config.py

# categories for the transaction classifier — kinda like the label dictionary
ID2LABEL = {
    0: "Shopping",
    1: "Dining Out",
    2: "Entertainment",
    3: "Transportation",
    4: "Housing",
    5: "Payments/Credits",
    6: "Utilities",
    7: "Service Subscriptions"
}

# doing the reverse mapping so we can go label -> id when needed
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# the base pretrained model we fine-tune (basically FinBERT from HF)
BASE_MODEL_NAME = "ProsusAI/finbert"

# where our fine-tuned model gets saved — app will load it from here later
FINE_TUNED_MODEL_PATH = "./my_finbert_classifier"

# folder where training stuff goes (checkpoints, logs, etc.)
TRAINING_OUTPUT_DIR = "./results"

# css file for styling the streamlit app UI
STYLE_CSS_PATH = "style.css"

# training hyperparameters — nothing crazy, just basic small-dataset values
TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "logging_steps": 50,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
}
