# ============================================================
# PIPELINE - PREDIÇÃO DE RISCO DE CRÉDITO
# FIAP Pós Tech - Data Analytics - Fase 2
# Grupo Ellen
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42
THRESHOLD = 0.10

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"

CREDIT_FILE = RAW_DIR / "credit_record.csv"
APPLICATION_FILE = RAW_DIR / "application_record.csv"


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

def load_data():

    df_credit = pd.read_csv(CREDIT_FILE)
    df_application = pd.read_csv(APPLICATION_FILE)

    print("Dados carregados:")
    print(f"credit_record: {df_credit.shape}")
    print(f"application_record: {df_application.shape}")

    return df_credit, df_application


# ============================================================
# 3. CRIAÇÃO DA TARGET
# ============================================================

def create_target(df_credit):

    # Conversão dos status para valores numéricos
    df_credit = df_credit.copy()

    df_credit["STATUS_NUM"] = (
        df_credit["STATUS"]
        .replace({
            "C": 0,
            "X": 0
        })
        .astype(int)
    )

    # Agrupamento por cliente
    target = (
        df_credit
        .groupby("ID")
        .agg(
            TOTAL_MESES=("STATUS_NUM", "count"),

            MESES_STATUS_0=(
                "STATUS_NUM",
                lambda x: (x == 0).sum()
            ),

            MESES_STATUS_1=(
                "STATUS_NUM",
                lambda x: (x == 1).sum()
            ),

            MESES_STATUS_2_MAIS=(
                "STATUS_NUM",
                lambda x: (x >= 2).sum()
            )
        )
        .reset_index()
    )

    # Regra de negócio 3
    target["TARGET"] = (
        (target["MESES_STATUS_1"] >= 2)
        |
        (target["MESES_STATUS_2_MAIS"] >= 1)
        |
        (target["MESES_STATUS_0"] >= 30)
    ).astype(int)

    print("\nDistribuição da TARGET:")
    print(target["TARGET"].value_counts())

    print("\nProporção da TARGET:")
    print(target["TARGET"].value_counts(normalize=True))

    return target


# ============================================================
# 4. PREPARAÇÃO DA BASE DE APPLICATION
# ============================================================

def prepare_application(df_application):

    df_application = df_application.copy()

    # --------------------------------------------------------
    # Tratamento de valores ausentes em OCCUPATION_TYPE
    # --------------------------------------------------------

    df_application["OCCUPATION_TYPE"] = (
        df_application["OCCUPATION_TYPE"]
        .fillna("Not reported")
    )

    # --------------------------------------------------------
    # Transformação de idade
    # --------------------------------------------------------

    df_application["AGE"] = (
        -df_application["DAYS_BIRTH"] / 365.25
    ).round(1)

    # --------------------------------------------------------
    # Tratamento de DAYS_EMPLOYED
    # --------------------------------------------------------

    df_application["DAYS_EMPLOYED_CLEAN"] = (
        df_application["DAYS_EMPLOYED"]
        .where(df_application["DAYS_EMPLOYED"] < 0)
    )

    # --------------------------------------------------------
    # Transformação de tempo de emprego em anos
    # Valores positivos representam pensionistas
    # --------------------------------------------------------

    df_application["YEARS_EMPLOYED"] = (
        -df_application["DAYS_EMPLOYED_CLEAN"] / 365.25
    ).fillna(60)

    return df_application


# ============================================================
# 5. MERGE E PREPARAÇÃO PARA MODELAGEM
# ============================================================

def prepare_model_data(df_application, target):

    # --------------------------------------------------------
    # Merge entre application_record e TARGET
    # --------------------------------------------------------

    df_model = df_application.merge(
        target[["ID", "TARGET"]],
        on="ID",
        how="inner"
    )

    print("\nBase após o merge:")
    print(df_model.shape)

    # --------------------------------------------------------
    # Remoção de variáveis não utilizadas
    # --------------------------------------------------------

    df_modela = df_model.drop(
        columns=[
            "ID",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_EMPLOYED_CLEAN",
            "FLAG_MOBIL"
        ]
    ).copy()

    # --------------------------------------------------------
    # Variáveis categóricas
    # --------------------------------------------------------

    categorical_vars = [
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE"
    ]

    # --------------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------------

    df_modelb = pd.get_dummies(
        df_modela,
        columns=categorical_vars,
        dtype="uint8"
    )

    # --------------------------------------------------------
    # Conversão das variáveis binárias
    # --------------------------------------------------------

    df_modelb["FLAG_OWN_CAR"] = (
        df_modelb["FLAG_OWN_CAR"]
        .map({"Y": 1, "N": 0})
    )

    df_modelb["FLAG_OWN_REALTY"] = (
        df_modelb["FLAG_OWN_REALTY"]
        .map({"Y": 1, "N": 0})
    )

    # --------------------------------------------------------
    # Transformação logarítmica da renda
    # --------------------------------------------------------

    df_modelb["AMT_INCOME_TOTAL_LOG"] = (
        np.log1p(df_model["AMT_INCOME_TOTAL"])
    )

    # Remoção da renda original
    df_modelc = df_modelb.drop(
        columns=["AMT_INCOME_TOTAL"]
    ).copy()

    # --------------------------------------------------------
    # Remoção de CNT_FAM_MEMBERS
    # devido à alta correlação com CNT_CHILDREN
    # --------------------------------------------------------

    df_modeld = df_modelc.drop(
        columns=["CNT_FAM_MEMBERS"]
    ).copy()

    print("\nBase final para modelagem:")
    print(df_modeld.shape)

    # --------------------------------------------------------
    # Separação entre X e y
    # --------------------------------------------------------

    X = df_modeld.drop(
        columns=["TARGET"]
    )

    y = df_modeld["TARGET"]

    return X, y


# ============================================================
# 6. DIVISÃO TREINO E TESTE
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDivisão dos dados:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 7. TREINAMENTO DO RANDOM FOREST
# ============================================================

def train_model(X_train, y_train):

    modelo_rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo_rf.fit(
        X_train,
        y_train
    )

    return modelo_rf


# ============================================================
# 8. AVALIAÇÃO DO MODELO
# ============================================================

def evaluate_model(modelo_rf, X_test, y_test):

    # Probabilidades
    y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # Threshold padrão = 0.50
    # --------------------------------------------------------

    y_pred_50 = (
        y_prob_rf >= 0.50
    ).astype(int)

    print("\n" + "=" * 60)
    print("AVALIAÇÃO - THRESHOLD 0.50")
    print("=" * 60)

    print(
        f"Accuracy:  {accuracy_score(y_test, y_pred_50):.4f}"
    )

    print(
        f"Precision: {precision_score(y_test, y_pred_50, zero_division=0):.4f}"
    )

    print(
        f"Recall:    {recall_score(y_test, y_pred_50):.4f}"
    )

    print(
        f"F1-Score:  {f1_score(y_test, y_pred_50):.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}"
    )

    print(
        f"PR-AUC:    {average_precision_score(y_test, y_prob_rf):.4f}"
    )

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred_50))

    # --------------------------------------------------------
    # Threshold escolhido = 0.10
    # --------------------------------------------------------

    y_pred = (
        y_prob_rf >= THRESHOLD
    ).astype(int)

    print("\n" + "=" * 60)
    print(f"AVALIAÇÃO - THRESHOLD {THRESHOLD:.2f}")
    print("=" * 60)

    print(
        f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall:    {recall_score(y_test, y_pred):.4f}"
    )

    print(
        f"F1-Score:  {f1_score(y_test, y_pred):.4f}"
    )

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    return y_prob_rf, y_pred


# ============================================================
# 9. IMPORTÂNCIA DAS VARIÁVEIS
# ============================================================

def feature_importance(modelo_rf, X_train):

    importancias = pd.DataFrame({
        "Variavel": X_train.columns,
        "Importancia": modelo_rf.feature_importances_
    })

    importancias = (
        importancias
        .sort_values(
            "Importancia",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("TOP 15 VARIÁVEIS MAIS IMPORTANTES")
    print("=" * 60)

    print(
        importancias.head(15).to_string(index=False)
    )

    return importancias


# ============================================================
# 10. EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("PIPELINE DE PREDIÇÃO DE RISCO DE CRÉDITO")
    print("=" * 60)

    # 1. Carregar dados
    df_credit, df_application = load_data()

    # 2. Criar target
    target = create_target(df_credit)

    # 3. Preparar application
    df_application = prepare_application(
        df_application
    )

    # 4. Preparar dados para modelagem
    X, y = prepare_model_data(
        df_application,
        target
    )

    # 5. Dividir treino e teste
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # 6. Treinar Random Forest
    modelo_rf = train_model(
        X_train,
        y_train
    )

    # 7. Avaliar modelo
    y_prob_rf, y_pred = evaluate_model(
        modelo_rf,
        X_test,
        y_test
    )

    # 8. Importância das variáveis
    importancias = feature_importance(
        modelo_rf,
        X_train
    )

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTADO COM SUCESSO!")
    print("=" * 60)

    return {
        "modelo": modelo_rf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_prob": y_prob_rf,
        "y_pred": y_pred,
        "importancias": importancias
    }


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()