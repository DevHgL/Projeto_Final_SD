# evaluate_test.py
# Uso:
#   py evaluate_test.py --test data/test.csv --target Class --model models/credit_lr_smote.joblib --threshold 0.5

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Caminho para data/test.csv")
    ap.add_argument("--target", default="Class", help="Nome da coluna alvo")
    ap.add_argument("--model", required=True, help="Caminho do modelo .joblib")
    ap.add_argument("--threshold", type=float, default=0.5, help="Limiar para classe 1")
    args = ap.parse_args()

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test)
    if args.target not in df.columns:
        raise ValueError(f"Coluna alvo '{args.target}' não encontrada em {args.test}")

    y_true = df[args.target].values
    X = df.drop(columns=[args.target])

    pipe = joblib.load(args.model)

    # probabilidade da classe 1 (suporta LR/XGB/etc. com predict_proba)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    elif hasattr(pipe, "decision_function"):
        # fallback para modelos sem predict_proba
        scores = pipe.decision_function(X)
        # normaliza sigmoid para virar pseudo-probabilidade
        proba = 1 / (1 + np.exp(-scores))
    else:
        # sem probabilidades
        proba = None

    if proba is not None:
        y_pred = (proba >= args.threshold).astype(int)
    else:
        y_pred = pipe.predict(X)

    # métricas
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
    }

    if proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc = roc_auc_score(y_true, proba)
            ap = average_precision_score(y_true, proba)  # PR-AUC
            metrics["roc_auc"] = roc
            metrics["pr_auc"] = ap
        except Exception:
            pass

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1])
    report_txt = classification_report(y_true, y_pred, digits=4)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    pd.DataFrame([metrics]).to_csv(out_dir / "test_metrics.csv", index=False)

    print("===== Métricas (TEST) =====")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}")
    print("\nRelatório de Classificação:\n")
    print(report_txt)
    print(f"\nArquivos salvos em {out_dir}/")
    print(" - test_metrics.csv")
    print(" - confusion_matrix.csv")

if __name__ == "__main__":
    main()
