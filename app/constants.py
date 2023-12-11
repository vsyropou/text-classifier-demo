"""Here we keep all constants in one place"""

SEED = 999

# These are the features that we will use for training
#  we exclude the id since that would be by definition overfitting
RAW_FEATURES = ["query"]

# This is the fature that we want to predict
TARGET_FEATURE = "target"


INPUT_LAYER_SPECS = [
    dict(klass="Conv1D", filters=2**3, kernel_size=2**2, activation="selu"),
    dict(klass="MaxPooling1D", pool_size=2),
]
OUTPUT_LAYER_SPECS = None  # [dict(units=16, activation="selu")]
OPTIMIZER_SPECS = {"name": "Adam", "learning_rate": 0.0001}
LOSS_SPECS = {"name": "CategoricalCrossentropy"}
