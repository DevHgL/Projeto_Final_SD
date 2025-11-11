# run.py
# Pipeline completo do projeto:
#  1) Preprocessamento
#  2) Treino do modelo
#  3) Avaliação no teste
#  4) Busca de melhor threshold
#  5) Geração de gráficos
#
# Execução:
#    python3 run.py
#

import subprocess
import sys
from pathlib import Path

def run(cmd):
    print("\n" + "="*60)
    print(f"Executando: {cmd}")
    print("="*60 + "\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Erro ao executar: {cmd}")
        sys.exit(result.returncode)

def main():
    root = Path(__file__).resolve().parent

    # garante pastas básicas
    for folder in ["data", "models", "reports"]:
        Path(folder).mkdir(exist_ok=True)

    # 1) PREPARAÇÃO DOS DADOS
    prep = root / "prep_credit_data.py"
    if prep.exists():
        run("python3 prep_credit_data.py")
    else:
        print("prep_credit_data.py não encontrado! ❌")
        sys.exit(1)

    # 2) TREINO DO MODELO
    train = root / "train_credit_model.py"
    if train.exists():
        run("python3 train_credit_model.py")
    else:
        print("train_credit_model.py não encontrado! ❌")
        sys.exit(1)

    # 3) AVALIAÇÃO NO TESTE
    evaluate = root / "evaluate_test.py"
    if evaluate.exists():
        run(
            "python3 evaluate_test.py "
            "--test data/test.csv "
            "--target Class "
            "--model models/credit_lr_smote.joblib"
        )
    else:
        print("evaluate_test.py não encontrado! ❌")
        sys.exit(1)

    # 4) BUSCA DO MELHOR THRESHOLD
    thr_file = root / "threshold_search.py"
    if thr_file.exists():
        run(
            "python3 threshold_search.py "
            "--file data/cleaned_full.csv "
            "--target Class "
            "--model models/credit_lr_smote.joblib "
            "--metric f1"
        )
    else:
        print("threshold_search.py não encontrado — pulando.")

    # 5) PLOTAGEM DOS GRÁFICOS
    plot_script = root / "plot_results.py"
    if plot_script.exists():
        run(
            "python3 plot_results.py "
            "--test data/test.csv "
            "--target Class "
            "--model models/credit_lr_smote.joblib "
            "--best_threshold reports/best_threshold.csv"
        )
    else:
        print("plot_results.py não encontrado — pulando.")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print("Gerados:")
    print(" - data/")
    print(" - models/")
    print(" - reports/ (inclui gráficos e métricas)")
    print("="*60)

if __name__ == "__main__":
    main()
