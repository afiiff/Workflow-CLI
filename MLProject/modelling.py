"""
modelling_Afiif_Alfarabi.py

Training Random Forest + MLflow Autolog
menggunakan dataset hasil preprocessing.

Input:

* train_preprocessed.csv
* test_preprocessed.csv
  """

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# ======================================================

# MLFLOW CONFIGURATION

# ======================================================

EXPERIMENT_NAME = "Heart_Disease_RF_Basic"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

# ======================================================

# LOAD DATASET

# ======================================================

BASE_DIR = os.path.dirname(
os.path.abspath(__file__)
)

TRAIN_PATH = os.path.join(
BASE_DIR,
"train_preprocessed.csv"
)

TEST_PATH = os.path.join(
BASE_DIR,
"test_preprocessed.csv"
)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Train Shape : {train_df.shape}")
print(f"Test Shape  : {test_df.shape}")

# ======================================================

# TARGET BINARIZATION

# ======================================================

train_df["target"] = (
train_df["num"] > 0
).astype(int)

test_df["target"] = (
test_df["num"] > 0
).astype(int)

train_df.drop(
columns=["num"],
inplace=True
)

test_df.drop(
columns=["num"],
inplace=True
)

# ======================================================

# SPLIT FEATURE & TARGET

# ======================================================

X_train = train_df.drop(
columns=["target"]
)

y_train = train_df["target"]

X_test = test_df.drop(
columns=["target"]
)

y_test = test_df["target"]

print("\nTarget Distribution (Train)")
print(y_train.value_counts())

# ======================================================

# MLFLOW AUTOLOG

# ======================================================

mlflow.sklearn.autolog(
log_input_examples=True,
log_model_signatures=True,
log_models=True,
silent=False
)

# ======================================================

# MODEL TRAINING

# ======================================================

with mlflow.start_run(
run_name="RandomForest_Afiif"
) as run:

    print(
        f"\nRun ID : {run.info.run_id}"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    
    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nHASIL EVALUASI")
    print("-" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

print("\nTraining selesai.")
print("Jalankan:")
print("mlflow ui --backend-store-uri mlruns")
