# train_credit_model.py
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore", category=UserWarning)

USE_SMOTE = True
SMOTE_OK = False
try:
    if USE_SMOTE:
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        SMOTE_OK = True
except Exception:
    SMOTE_OK = False

DATA_DIR = Path("data")
TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"
TARGET = "Class"

def load_xy(path: Path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET}' não encontrada em {path}")
    y = df[TARGET].values
    X = df.drop(columns=[TARGET]).values
    return X, y

def main():
    if not TRAIN.exists():
        print("Arquivo data/train.csv não encontrado. Rode antes: py prep_credit_data.py")
        sys.exit(1)

    X_train, y_train = load_xy(TRAIN)

    print(f"Treino: X={X_train.shape}, y pos={y_train.sum()} ({y_train.mean():.4%})")

    # Modelo base
    clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)

    if SMOTE_OK:
        print("Usando SMOTE + LogisticRegression")
        pipe = Pipeline(steps=[
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("clf", clf),
        ])
        model = pipe
    else:
        print("imblearn/SMOTE indisponível. Usando apenas class_weight='balanced'.")
        model = clf

    # CV para ter métricas mais estáveis
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_train, y_train, cv=cv, method="predict")
    try:
        y_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        roc = roc_auc_score(y_train, y_proba)
    except Exception:
        roc = np.nan

    bal_acc = balanced_accuracy_score(y_train, y_pred)
    print("\n===== Métricas (CV=5) =====")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    if not np.isnan(roc):
        print(f"ROC-AUC:          {roc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_train, y_pred, digits=4))

    # Treina final no conjunto todo e salva
    model.fit(X_train, y_train)
    import joblib
    MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "credit_lr_smote.joblib")
    print("\nModelo salvo em: models/credit_lr_smote.joblib")

if __name__ == "__main__":
    main()
