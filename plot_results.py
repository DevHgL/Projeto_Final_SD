# plot_results.py
# Gera gráficos: ROC, PR Curve, Confusion Matrix, Histograma de Scores

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

def load_best_threshold(path):
    path = Path(path)
    if not path.exists():
        return 0.5
    df = pd.read_csv(path)

    # tenta encontrar a coluna do best threshold
    for col in ["best_threshold", "threshold", "thr"]:
        if col in df.columns:
            try:
                thr = float(df[col].iloc[0])
                return float(np.clip(thr, 0, 1))
            except:
                pass

    # fallback: tenta achar qualquer número entre 0 e 1
    vals = df.select_dtypes(include=[np.number]).values.flatten()
    vals = [v for v in vals if 0 <= v <= 1]
    return float(vals[0]) if vals else 0.5


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    return model.predict(X).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", default="Class")
    ap.add_argument("--model", required=True)
    ap.add_argument("--best_threshold", default=None)
    args = ap.parse_args()

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    # =======================
    # Carrega dados e modelo
    # =======================
    df = pd.read_csv(args.test)
    y = df[args.target].values
    X = df.drop(columns=[args.target]).values

    model = joblib.load(args.model)
    scores = get_scores(model, X)

    # Se tiver threshold, carrega
    thr = 0.5
    if args.best_threshold:
        thr = load_best_threshold(args.best_threshold)

    # ==================================
    # Gráfico ROC
    # ==================================
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("FPR (1 - Especificidade)")    # <- corrigido
    plt.ylabel("TPR (Recall)")
    plt.title("Curva ROC")
    plt.legend()
    roc_path = out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ==================================
    # Precision-Recall
    # ==================================
    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    pr_path = out_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ==================================
    # Histograma de scores
    # ==================================
    plt.figure()
    plt.hist(scores[y == 0], bins=40, alpha=0.6, label="Classe 0")
    plt.hist(scores[y == 1], bins=40, alpha=0.6, label="Classe 1")
    plt.axvline(thr, color="red", linestyle="--", lw=2, label=f"Threshold = {thr:.4f}")
    plt.xlabel("Score (probabilidade)")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos Scores")
    plt.legend()
    hist_path = out_dir / "scores_hist.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ==================================
    # Matriz de Confusão
    # ==================================
    preds = (scores >= thr).astype(int)
    cm = confusion_matrix(y, preds, labels=[0,1])

    plt.figure()
    disp = ConfusionMatrixDisplay(cm, display_labels=["0","1"])
    disp.plot(values_format="d")
    plt.title(f"Matriz de Confusão (thr={thr:.4f})")
    cm_path = out_dir / "confusion_matrix_thr.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n✅ Gráficos gerados em reports/:")
    print(f" - {roc_path.name}")
    print(f" - {pr_path.name}")
    print(f" - {hist_path.name}")
    print(f" - {cm_path.name}")


if __name__ == "__main__":
    main()
