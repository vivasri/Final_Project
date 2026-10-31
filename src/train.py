"""
Model Training & Benchmarking Pipeline
=====================================
Project: AI-Driven Climate Risk Forecasting & Adaptive Decision Support System (Tamil Nadu MVP)

Implements:
1. Baselines:
   - Climatological Mean (predicts 0 anomaly)
   - Persistence (predicts lag-1 anomaly)
2. Machine Learning Models:
   - Random Forest Regressor & Classifier
   - Regularized XGBoost Regressor & Classifier
3. Sequential Deep Learning Model:
   - Long Short-Term Memory (LSTM) sequence-to-point network
4. Honest Benchmarking & Metric Reporting:
   - RMSE, MAE, R2, Pearson correlation
   - Multiclass Accuracy, Precision, Recall, Macro-F1
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, target_name: str) -> dict:
    """
    Compute RMSE, MAE, R2, and Pearson correlation coefficient.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate correlation (if predictions have variance)
    if np.std(y_pred) > 1e-6 and np.std(y_true) > 1e-6:
        corr, _ = pearsonr(y_true, y_pred)
    else:
        corr = 0.0

    return {
        "model": model_name,
        "target": target_name,
        "rmse": round(float(rmse), 3),
        "mae": round(float(mae), 3),
        "r2": round(float(r2), 4),
        "pearson_r": round(float(corr), 4)
    }


def train_baselines(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    lag_col: str,
    target_label: str
) -> list[dict]:
    """
    Evaluate Climatology and Persistence baselines.
    """
    y_test = df_test[target_col].values

    # 1. Climatology Baseline: predicts anomaly = 0
    y_pred_clim = np.zeros_like(y_test)
    clim_metrics = calculate_regression_metrics(y_test, y_pred_clim, "Climatology Baseline", target_label)

    # 2. Persistence Baseline: predicts y_hat(t) = y(t-1)
    # The anomaly equivalent of persistence is prior month's anomaly
    y_pred_pers = (df_test[lag_col] - df_test["train_clim_rain"]).fillna(0).values if "rain" in target_col else (df_test[lag_col] - df_test["train_clim_temp"]).fillna(0).values
    pers_metrics = calculate_regression_metrics(y_test, y_pred_pers, "Persistence Baseline", target_label)

    return [clim_metrics, pers_metrics]


def train_tree_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_label: str,
    model_prefix: str
) -> tuple[RandomForestRegressor, XGBRegressor, list[dict]]:
    """
    Train and evaluate Random Forest and XGBoost regressors.
    """
    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_val = df_val[feature_cols].values
    y_val = df_val[target_col].values

    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    # 1. Random Forest Regressor
    logging.info(f"Training Random Forest Regressor for {target_label}...")
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = calculate_regression_metrics(y_test, rf_pred, "Random Forest", target_label)

    # 2. XGBoost Regressor
    logging.info(f"Training XGBoost Regressor for {target_label}...")
    xgb = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = calculate_regression_metrics(y_test, xgb_pred, "XGBoost", target_label)

    # Save trained models
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, f"models/rf_{model_prefix}_model.joblib")
    joblib.dump(xgb, f"models/xgb_{model_prefix}_model.joblib")

    return rf, xgb, [rf_metrics, xgb_metrics]


def build_and_train_lstm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_label: str
) -> tuple[object, dict]:
    """
    Train an LSTM sequence-to-point network using PyTorch if installed.
    Falls back gracefully if PyTorch is still initializing.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(42)
        device = torch.device("cpu")

        # Prepare temporal sequences per district (sequence length = 3 months)
        seq_len = 3
        
        def make_sequences(df):
            X_seqs, y_seqs = [], []
            for _, grp in df.groupby("district"):
                grp = grp.sort_values("date")
                feats = grp[feature_cols].values
                targets = grp[target_col].values
                for i in range(len(grp) - seq_len + 1):
                    X_seqs.append(feats[i : i + seq_len])
                    y_seqs.append(targets[i + seq_len - 1])
            return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)

        X_train_seq, y_train_seq = make_sequences(df_train)
        X_val_seq, y_val_seq = make_sequences(df_val)
        X_test_seq, y_test_seq = make_sequences(df_test)

        train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        class ClimateLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=32, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                )

            def forward(self, x):
                out, (hn, cn) = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out.squeeze(-1)

        input_dim = len(feature_cols)
        model = ClimateLSTM(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        # Train model
        model.train()
        for epoch in range(40):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_preds = model(torch.from_numpy(X_test_seq)).numpy()
            
        metrics = calculate_regression_metrics(y_test_seq, test_preds, "LSTM Recurrent", target_label)
        torch.save(model.state_dict(), "models/lstm_rain_model.pt")
        logging.info("Saved trained LSTM to models/lstm_rain_model.pt")
        return model, metrics

    except Exception as e:
        logging.warning(f"PyTorch LSTM training skipped or deferred ({e}).")
        return None, None


def train_risk_classifiers(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str]
) -> tuple[RandomForestClassifier, XGBClassifier, dict]:
    """
    Train and benchmark multiclass hazard risk classifiers (Low, Moderate, High, Severe).
    """
    logging.info("Training Hazard Risk Classifiers (Low, Moderate, High, Severe)...")
    X_train = df_train[feature_cols].values
    y_train = df_train["target_risk_class"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["target_risk_class"].values

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average="macro", zero_division=0)

    # XGBoost Classifier
    xgb_clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train, y_train)
    xgb_preds = xgb_clf.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_preds, average="macro", zero_division=0)

    joblib.dump(rf_clf, "models/rf_risk_classifier.joblib")
    joblib.dump(xgb_clf, "models/xgb_risk_classifier.joblib")

    clf_metrics = {
        "Random Forest Classifier": {
            "accuracy": round(float(rf_acc), 4),
            "macro_precision": round(float(rf_prec), 4),
            "macro_recall": round(float(rf_rec), 4),
            "macro_f1": round(float(rf_f1), 4)
        },
        "XGBoost Classifier": {
            "accuracy": round(float(xgb_acc), 4),
            "macro_precision": round(float(xgb_prec), 4),
            "macro_recall": round(float(xgb_rec), 4),
            "macro_f1": round(float(xgb_f1), 4)
        }
    }
    return rf_clf, xgb_clf, clf_metrics


def run_training_pipeline():
    """
    Execute full Phase 3 model training, evaluation, and benchmark generation.
    """
    # 1. Load preprocessed feature splits
    df_train = pd.read_csv("data/processed/train_features.csv")
    df_val = pd.read_csv("data/processed/val_features.csv")
    df_test = pd.read_csv("data/processed/test_features.csv")

    with open("data/processed/split_metadata.json") as f:
        meta = json.load(f)
    feature_cols = [f"{col}_scaled" for col in meta["feature_cols"]]

    # 2. Benchmark Models for Target 1: Rainfall Anomaly
    logging.info("=== Benchmarking Models for Target: Rainfall Anomaly ===")
    rain_baselines = train_baselines(df_train, df_val, df_test, "target_rain_anomaly", "rain_lag1", "Rain Anomaly (mm)")
    rf_rain, xgb_rain, rain_trees = train_tree_models(
        df_train, df_val, df_test, feature_cols, "target_rain_anomaly", "Rain Anomaly (mm)", "rain"
    )
    lstm_model, lstm_metrics = build_and_train_lstm(
        df_train, df_val, df_test, feature_cols, "target_rain_anomaly", "Rain Anomaly (mm)"
    )

    all_rain_metrics = rain_baselines + rain_trees
    if lstm_metrics:
        all_rain_metrics.append(lstm_metrics)

    # 3. Benchmark Models for Target 2: Temperature Anomaly
    logging.info("=== Benchmarking Models for Target: Temperature Anomaly ===")
    temp_baselines = train_baselines(df_train, df_val, df_test, "target_temp_anomaly", "temp_lag1", "Temp Anomaly (C)")
    rf_temp, xgb_temp, temp_trees = train_tree_models(
        df_train, df_val, df_test, feature_cols, "target_temp_anomaly", "Temp Anomaly (C)", "temp"
    )
    all_temp_metrics = temp_baselines + temp_trees

    # 4. Train Multiclass Hazard Risk Classifiers
    rf_clf, xgb_clf, clf_metrics = train_risk_classifiers(df_train, df_val, df_test, feature_cols)

    # 5. Compile and Save Benchmark Report
    df_metrics = pd.DataFrame(all_rain_metrics + all_temp_metrics)
    os.makedirs("reports", exist_ok=True)
    df_metrics.to_csv("reports/benchmark_metrics.csv", index=False)

    with open("reports/classification_metrics.json", "w") as f:
        json.dump(clf_metrics, f, indent=2)

    return df_metrics, clf_metrics


if __name__ == "__main__":
    df_metrics, clf_metrics = run_training_pipeline()
    print("\n" + "=" * 70)
    print("PHASE 3 MODEL BENCHMARKING REPORT")
    print("=" * 70)
    print("\nContinuous Anomaly Forecasting Metrics (Unseen Test Partition: 2024-2025):")
    print(df_metrics.to_string(index=False))

    print("\nHazard Risk Classification Metrics (Unseen Test Partition):")
    for model_name, m in clf_metrics.items():
        print(f"  {model_name}:")
        print(f"    Accuracy:  {m['accuracy'] * 100:.2f}%")
        print(f"    Macro F1:  {m['macro_f1']:.4f}")
        print(f"    Precision: {m['macro_precision']:.4f}, Recall: {m['macro_recall']:.4f}")
    print("=" * 70)
