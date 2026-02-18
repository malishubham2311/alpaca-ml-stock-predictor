import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from Data_fetch import fetch_stock_bars
from feature_engineer import engineer_features
import joblib


def prepare_data_cls(symbol="AAPL", days_back=1095):
    df_raw = fetch_stock_bars(symbol, days_back=days_back)
    df = engineer_features(df_raw)

    # Use classification target
    target_col = "target_cls"
    drop_cols = [
        "symbol", "timestamp",
        "open", "high", "low", "close", "volume",
        "target", "next_return",  # regression targets
        target_col,
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df[target_col].values

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler, feature_cols


def evaluate_cls(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = np.nan

    print(f"\n{name}")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    return {"accuracy": acc, "f1": f1, "auc": auc}


def train_cls_models():
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data_cls()

    results = {}
    models = {}

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    results["RandomForestClassifier"] = evaluate_cls("RandomForestClassifier", rf, X_test, y_test)
    models["rf_cls"] = rf

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    results["GradientBoostingClassifier"] = evaluate_cls("GradientBoostingClassifier", gb, X_test, y_test)
    models["gb_cls"] = gb

    model_data = {
        "results": results,
        "models": models,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
    joblib.dump(model_data, "cls_aapl_model.joblib")
    print("\n✅ Classification model saved: cls_aapl_model.joblib")

    return model_data


if __name__ == "__main__":
    train_cls_models()
