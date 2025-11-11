# prep_credit_data.py
# Execute simplesmente:
#   python prep_credit_data.py
# ou
#   py prep_credit_data.py
#
# Saídas:
#   ./data/cleaned_full.csv
#   ./data/train.csv
#   ./data/test.csv
#   Métricas no console (quick fit)

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

# ========= Defaults desejados =========
DEFAULTS = dict(
    csv_path="creditcard.csv",
    target="Class",
    outlier_method="isoforest",  # none | iqr | isoforest
    iqr_factor=1.5,
    iso_contamination="auto",    # "auto" ou float, ex.: 0.02
    scaler="robust",             # standard | robust | minmax
    test_size=0.2,
    random_state=42,
    quick_fit=True
)

def choose_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError("Scaler inválido. Use: standard | robust | minmax")

def load_csv(csv_path: str) -> pd.DataFrame:
    print(f"Carregando CSV: {csv_path}")
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path}\n"
            f"Dica: coloque o arquivo no diretório atual ou passe --csv_path /caminho/arquivo.csv"
        )
    return pd.read_csv(p)

def fill_basic_nans(df: pd.DataFrame) -> pd.DataFrame:
    # Remove colunas 100% vazias
    df = df.dropna(axis=1, how="all")
    # Preenche NaNs *básico* (reforçado depois por SimpleImputer na pipeline)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
    return df

def remove_outliers_isoforest(df: pd.DataFrame, feature_cols: list, contamination="auto", seed=42) -> pd.DataFrame:
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1
    )
    preds = iso.fit_predict(df[feature_cols])
    mask = preds == 1
    removed = (~mask).sum()
    print(f"Outliers removidos (IsolationForest): {removed} de {len(mask)}")
    return df.loc[mask].reset_index(drop=True)

def remove_outliers_iqr(df: pd.DataFrame, feature_cols: list, k=1.5) -> pd.DataFrame:
    q1 = df[feature_cols].quantile(0.25)
    q3 = df[feature_cols].quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    mask = ((df[feature_cols] >= low) & (df[feature_cols] <= high)).all(axis=1)
    removed = (~mask).sum()
    print(f"Outliers removidos (IQR k={k}): {removed} de {len(mask)}")
    return df.loc[mask].reset_index(drop=True)

def main():
    # Parser mantém argumentos, mas com defaults prontos
    ap = argparse.ArgumentParser(description="Tratamento de CSV para treino de IA (defaults prontos).")
    ap.add_argument("--csv_path", type=str, default=DEFAULTS["csv_path"])
    ap.add_argument("--target", type=str, default=DEFAULTS["target"])
    ap.add_argument("--outlier_method", type=str, default=DEFAULTS["outlier_method"],
                    choices=["none", "iqr", "isoforest"])
    ap.add_argument("--iqr_factor", type=float, default=DEFAULTS["iqr_factor"])
    ap.add_argument("--iso_contamination", type=str, default=DEFAULTS["iso_contamination"])
    ap.add_argument("--scaler", type=str, default=DEFAULTS["scaler"], choices=["standard", "robust", "minmax"])
    ap.add_argument("--test_size", type=float, default=DEFAULTS["test_size"])
    ap.add_argument("--random_state", type=int, default=DEFAULTS["random_state"])
    ap.add_argument("--quick_fit", action="store_true", default=DEFAULTS["quick_fit"])
    args = ap.parse_args()

    # 1) Carrega
    df = load_csv(args.csv_path)
    print(f"Shape original: {df.shape}")

    # 2) Limpeza básica (garante que IForest não quebre com NaN)
    df = fill_basic_nans(df)

    # 3) Define alvo e features numéricas
    target_col = args.target if args.target in df.columns else None
    if target_col is None:
        print(f"[AVISO] Coluna alvo '{args.target}' não encontrada. Seguindo sem alvo.")
    X_all = df.drop(columns=[target_col]) if target_col else df.copy()
    X_all = X_all.select_dtypes(include=[np.number])  # apenas numéricos
    feature_cols = X_all.columns.tolist()

    # 4) Outliers
    if args.outlier_method == "isoforest" and feature_cols:
        contamination = args.iso_contamination
        if contamination != "auto":
            try:
                contamination = float(contamination)
            except ValueError:
                contamination = "auto"
        df = remove_outliers_isoforest(df, feature_cols, contamination=contamination, seed=args.random_state)
    elif args.outlier_method == "iqr" and feature_cols:
        df = remove_outliers_iqr(df, feature_cols, k=args.iqr_factor)
    else:
        print("[Outliers] Nenhuma remoção aplicada.")

    # 5) Re-seleciona X/y após remoção de outliers
    if target_col:
        y = df[target_col].values
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    else:
        y = None
        X = df.select_dtypes(include=[np.number])

    # 6) Split
    strat = y if (y is not None and len(np.unique(y)) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=strat
    )

    # 7) Pipeline com Imputer + Scaler + Modelo
    scaler = choose_scaler(args.scaler)
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])

    # 8) Treino rápido (mostra métricas)
    if y is not None and args.quick_fit:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print("\n===== Quick Fit: LogisticRegression (balanced) =====")
        print(classification_report(y_test, preds, digits=4))

    # 9) Salvar datasets já imputados + escalados (fit no treino, transforma treino/teste/todo)
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ajusta apenas com X_train (boa prática)
    fitted = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])
    fitted.fit(X_train)

    X_train_t = fitted.transform(X_train)
    X_test_t = fitted.transform(X_test)
    X_all_t = fitted.transform(X)  # transforma todo o conjunto para export "cleaned_full"

    # Reconstrói DataFrames com os mesmos nomes de colunas
    train_df = pd.DataFrame(X_train_t, columns=X.columns, index=pd.RangeIndex(len(X_train)))
    test_df = pd.DataFrame(X_test_t, columns=X.columns, index=pd.RangeIndex(len(X_test)))
    full_df = pd.DataFrame(X_all_t, columns=X.columns, index=pd.RangeIndex(len(X)))

    if y is not None:
        train_df[args.target] = y_train
        test_df[args.target] = y_test
        full_df[args.target] = y

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    full_df.to_csv(out_dir / "cleaned_full.csv", index=False)

    print(f"\nArquivos salvos em: {out_dir}/")
    print(" - cleaned_full.csv")
    print(" - train.csv")
    print(" - test.csv")

if __name__ == "__main__":
    main()
