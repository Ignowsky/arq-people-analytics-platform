import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, PrecisionRecallDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from lifelines import coxPHFitter

try:
    from .logger import setup_logger
    from .ml_preprocessing import drop_leakage_columns, split_train_test, build_preprocessor
except ImportError:
    from logger import setup_logger
    from ml_preprocessing import drop_leakage_columns, split_train_test, build_preprocessor

logger = setup_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    A Prova Real: Gera o relatório consolidado de Treino vs Teste.
    """
    logger.info("INICIANDO: Auditoria do Modelo (Relatório Consolidado)...")

    # 1. Validação Cruzada (Estabilidade do Modelo)
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='recall')
    logger.info(
        f"Recall Médio CV: {np.round(cv_scores.mean() * 100, 2)}% | Desvio: {np.round(cv_scores.std() * 100, 2)}")

    # 2. Previsões para as duas bases
    y_pred_train = model.predict(X_train)
    
    # Diminuindo a limiar de decisão para aumentar o recall (ajustado para 0.39 com base na curva ROC)
    probabilidade_testes = model.predict_proba(X_test)[:, 1]
    
    y_pred_test_custo = (probabilidade_testes >= 0.39).astype(int)

    # ---------------------------------------------------------
    # 🔥 A PROVA DO RECALL (RELATÓRIO DE MÉTRICAS CONSOLIDADO)
    # ---------------------------------------------------------
    print('\n' + '=' * 60)
    print('--- A PROVA DO RECALL (BASE DE TREINO) ---')
    print('=' * 60)
    print(classification_report(y_train, y_pred_train))

    print('\n' + '=' * 60)
    print('--- A PROVA DO RECALL (RELATÓRIO DE MÉTRICAS CONSOLIDADO) ---')
    print('=' * 60)
    print(classification_report(y_test, y_pred_test_custo))
    print('=' * 60)

    # 3. Gerando a Matriz de Confusão Visual (Base de Teste)
    cm = confusion_matrix(y_test, y_pred_test_custo)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=False,
                annot_kws={"size": 16, "weight": "bold"}, ax=ax)

    ax.set_xticklabels(['Retido (0)', 'Evasão (1)'], fontsize=12)
    ax.set_yticklabels(['Retido (0)', 'Evasão (1)'], fontsize=12, rotation=0)
    ax.set_title('Matriz de Confusão - Produção', fontsize=14, pad=20, weight='bold')

    plt.tight_layout()

    pasta_logs = os.path.join(BASE_DIR, "Logs")
    os.makedirs(pasta_logs, exist_ok=True)  # Cria a pasta se não existir

    caminho_imagem = os.path.join(pasta_logs, 'Matriz_Confusao_Final.png')
    fig.savefig(caminho_imagem, bbox_inches='tight', dpi=300)
    plt.close(fig)

    logger.info(f"SUCESSO: Auditoria finalizada e gráficos salvos em {pasta_logs}")


def run_training(df):
    logger.info("INICIANDO: Orquestração do Treinamento com Features Campeãs...")

    # 1. Limpeza inicial
    df_clean = drop_leakage_columns(df)

    # 2. Engenharia das Flags Binárias (Mimetizando o seu filtro)
    # Nota: Removi o ponto final do nome da variável pra evitar erro de sintaxe no Python
    df_clean['is_perfil_Nao_Mapeado'] = np.where(
        df_clean['perfil_comportamental'].str.contains('Não Mapeado', na=False), 1, 0
    )
    df_clean['is_dep_RELACIONAMENTO'] = np.where(
        df_clean['departamento_nome_api'] == 'RELACIONAMENTO', 1, 0
    )

    # Ajuste do caminho absoluto pra salvar a IA
    pasta_modelos = os.path.join(BASE_DIR, "Models")
    os.makedirs(pasta_modelos, exist_ok=True)

    # 3. A LISTA SAGRADA (Sincronizada com seu pedido)
    features_campeas = [
        'meses_de_casa',
        'salario_contratual',
        'idade',
        'qtd_dependentes',
        'is_perfil_Nao_Mapeado',
        'is_dep_RELACIONAMENTO'
    ]

    X = df_clean[features_campeas]
    y = df_clean['target_pediu_demissao']

    # 4. Split e Pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline_final = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'))
    ])

    pipeline_final.fit(X_train, y_train)

    # Invocando o preprocessor com o StandardScaler que acabamos de configurar
    # Adicionando o parâmetro k_neighbors = 2 para o SMOTE, para diminuir a alucinação sintética
    preprocessor = build_preprocessor()
    smote = SMOTE(random_state=42, k_neighbors = 2)

    # Regressão Logística com pesos balanceados
    # Adicionando o parâmetro C para controle de regularização (ajustado para evitar overfitting)
    lr_campeao = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', C = 0.1)

    # O Pipeline orquestra a ordem dos Jutsus
    pipeline_final = Pipeline(steps=[
        ('preprocessor', preprocessor),  # 1° Normaliza
        ('smote', smote),  # 2° Balanceia as classes
        ('classifier', lr_campeao)  # 3° Treina
    ])

    logger.info("Iniciando o treinamento do modelo (Fit) com StandardScaler...")
    pipeline_final.fit(X_train, y_train)
    logger.info("Treinamento concluído com sucesso!")

    # ---------------------------------------------------------
    # MUDANÇA TÁTICA: SALVAR O MODELO ANTES DE GERAR OS GRÁFICOS
    # ---------------------------------------------------------
    caminho_modelo = os.path.join(pasta_modelos, "lr_turnover_model.pkl")
    joblib.dump(pipeline_final, caminho_modelo)
    logger.info(f"SUCESSO ABSOLUTO: Modelo salvo em: {caminho_modelo}")

    # Agora sim, fazemos a auditoria visual
    evaluate_model(pipeline_final, X_train, y_train, X_test, y_test)
    
    # ----------------------------------------------------------
    # Adicionando a curva Recall-Precision para a base de teste
    # ----------------------------------------------------------
    
    logger.info("Gerando a Curva Recall-Precision para a base de teste...")
    pasta_logs = os.path.join(BASE_DIR, "Logs")
    fig_pr, ax_pr = plt.subplots(figsize = (7,5))
    
    # Usando o Scikit-learn para plotar a curva
    display = PrecisionRecallDisplay.from_estimator(
        pipeline_final,
        X_test,
        y_test,
        name = "Regressão Logística (SMOTE)",
        ax = ax_pr,
        color = "crimson",
        linewidth = 2
    )
    
    ax_pr.set_title("Curva Recall-Precision - Base de Teste", fontsize = 14, pad = 20, weight = "bold")
    plt.tight_layout()
    
    # Salvando a curva no mesmo diretório dos logs
    caminho_curva_pr = os.path.join(pasta_logs, "Curva_Recall_Precision.png")
    fig_pr.savefig(caminho_curva_pr, bbox_inches = "tight", dpi = 300)
    
    plt.close(fig_pr)
    logger.info(f"Curva Recall-Precision salva com sucesso em: {caminho_curva_pr}")

    return pipeline_final