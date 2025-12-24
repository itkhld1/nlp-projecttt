# config.py

# --- 1. Label and Model Configuration ---
# Define the categories for financial transaction classification.
ID2LABEL = {
    0: 'Shopping',
    1: 'Dining Out',
    2: 'Entertainment',
    3: 'Transportation',
    4: 'Housing',
    5: 'Payments/Credits',
    6: 'Utilities',
    7: 'Service Subscriptions'
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# --- 2. Path and Directory Configuration ---
# Specify the pre-trained model to be used as the base for fine-tuning.
BASE_MODEL_NAME = "ProsusAI/finbert"

# Define the local directory where the fine-tuned model will be saved and loaded from.
FINE_TUNED_MODEL_PATH = "./my_finbert_classifier"

# Specify the directory for storing training outputs, such as checkpoints.
TRAINING_OUTPUT_DIR = "./results"

# Specify the path to the CSS file for the Streamlit application.
STYLE_CSS_PATH = "style.css"

# --- 3. Training Hyperparameters ---
# These arguments control the fine-tuning process.
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
