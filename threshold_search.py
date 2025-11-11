# threshold_search.py
# Busca o melhor limiar (threshold) para um modelo binário maximizando F1 (ou outra métrica).
# Uso:
#   python3 threshold_search.py --file data/cleaned_full.csv --target Class \
#       --model models/credit_lr_smote.joblib --metric f1 --k 0.01
#
# Métricas suportadas:
#   - f1               : F1 no conjunto completo
#   - pr_auc           : Área PR AUC no conjunto completo
#   - pr_auc_at_k      : Área sob a curva Precisão-Recall até top-k%% das maiores probabilidades
#                        (ex.: k=0.01 => top 1%%)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split

def pr_auc_at_k(y_true, y_score, k=0.01):
    """PR-AUC considerando apenas o top-k%% (k em [0,1])."""
    n = len(y_score)
    top = max(1, int(np.ceil(k * n)))
    idx = np.argsort(-y_score)[:top]
    return average_precision_score(y_true[idx], y_score[idx])

def main():
    ap = argparse.ArgumentParser(description="Busca de threshold ideal para modelo binário.")
    ap.add_argument("--file", required=True, type=str, help="CSV já tratado (ex.: data/cleaned_full.csv).")
    ap.add_argument("--target", required=True, type=str, help="Nome da coluna alvo (ex.: Class).")
    ap.add_argument("--model", required=True, type=str, help="Caminho do modelo .joblib.")
    ap.add_argument("--metric", type=str, default="f1",
                    choices=["f1", "pr_auc", "pr_auc_at_k"],
                    help="Métrica a otimizar: f1 | pr_auc | pr_auc_at_k")
    # ATENÇÃO: em help strings do argparse, '%' precisa ser '%%'
    ap.add_argument("--k", type=float, default=0.01,
                    help="k em porcentagem para pr_auc_at_k (ex.: 0.01 = top 1%%)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Tamanho do holdout p/ avaliar o threshold.")
    ap.add_argument("--random_state", type=int, default=42, help="Seed.")
    args = ap.parse_args()

    csv_path = Path(args.file)
    model_path = Path(args.model)
    assert csv_path.exists(), f"Arquivo não encontrado: {csv_path}"
    assert model_path.exists(), f"Modelo não encontrado: {model_path}"

    # Carrega dados
    df = pd.read_csv(csv_path)
    y = df[args.target].values
    X = df.drop(columns=[args.target])

    # Divide para escolher threshold sem vazar informação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Carrega modelo treinado
    model = load(model_path)

    # Probabilidades no conjunto de validação
    if hasattr(model, "predict_proba"):
        val_scores = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        # normaliza decision_function para [0,1]
        df_raw = model.decision_function(X_val)
        df_min, df_max = df_raw.min(), df_raw.max()
        val_scores = (df_raw - df_min) / (df_max - df_min + 1e-12)
    else:
        raise ValueError("Modelo não possui predict_proba/decision_function.")

    # Grade de thresholds
    thresholds = np.linspace(0.01, 0.99, 99)

    best_thr, best_metric = 0.5, -np.inf
    for thr in thresholds:
        preds = (val_scores >= thr).astype(int)
        if args.metric == "f1":
            m = f1_score(y_val, preds, zero_division=0)
        elif args.metric == "pr_auc":
            m = average_precision_score(y_val, val_scores)
        else:  # pr_auc_at_k
            m = pr_auc_at_k(y_val, val_scores, k=args.k)

        if m > best_metric:
            best_metric = m
            best_thr = thr

    print("\n===== Threshold Search =====")
    print(f"Métrica        : {args.metric}")
    if args.metric == "pr_auc_at_k":
        print(f"k              : {args.k:.4f} (top {args.k*100:.2f}%)")
    print(f"Melhor thr     : {best_thr:.4f}")
    print(f"Melhor métrica : {best_metric:.6f}")

    # Salva relatório
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "best_threshold.csv"
    pd.DataFrame([{
        "metric": args.metric,
        "k": args.k if args.metric == "pr_auc_at_k" else np.nan,
        "best_threshold": best_thr,
        "best_metric": best_metric
    }]).to_csv(out_path, index=False)
    print(f"\nArquivo salvo: {out_path}")

if __name__ == "__main__":
    main()
