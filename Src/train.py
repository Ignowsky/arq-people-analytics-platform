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
from lifelines import CoxPHFitter


try:
    from .logger import setup_logger
    from .MlPreProcessing import split_train_test, build_preprocessor
    from .DataCleaning import drop_leakage_columns
except ImportError:
    from logger import setup_logger
    from MlPreProcessing import split_train_test, build_preprocessor
    from DataCleaning import drop_leakage_columns

logger = setup_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_survival_model(cph, df_test):
    """
    Avalia a precisão de tempo
    :param cph:
    :param df_test:
    :return:
    """

    logger.info("[INICIANDO]: Inicio do novo modelo de sobrevivência")

    # Utilizando o Concordance Index (c-index)
    # 0.5 = Chute / 1.0 = Acerto
    c_index = cph.concordance_index_
    logger.info(f"Concordance Index (Estabilidade): {np.round(c_index, 2)}")

    print('\n' + '=' * 60)
    print('--- RAIO-X das VARIÁVEIS (COX MODEL SUMMARY) ---')
    print('=' * 60)

    # visualizando o impacto real de cada feature no tempo de permanência
    print(cph.print_summary())
    print('=' * 60)

    # ---------------------------------------------------------
    # 🎨 AUDITORIA VISUAL (GERAÇÃO DOS GRÁFICOS)
    # ---------------------------------------------------------
    pasta_logs = os.path.join(BASE_DIR, "Logs")
    os.makedirs(pasta_logs, exist_ok=True)

    # GRÁFICO 1: Forest Plot (O Peso das Features)
    logger.info("Gerando gráfico de impacto das variáveis (Forest Plot)...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    cph.plot(ax=ax1)  # O lifelines faz a mágica sozinha aqui
    ax1.set_title('Impacto no Risco de Turnover (Hazard Ratios)', fontsize=14, weight='bold')
    plt.tight_layout()
    caminho_fig1 = os.path.join(pasta_logs, 'Cox_Impacto_Variaveis.png')
    fig1.savefig(caminho_fig1, bbox_inches='tight', dpi=300)
    plt.close(fig1)

    # GRÁFICO 2: A Curva de Sobrevivência Média
    logger.info("Gerando a Curva de Sobrevivência Média da empresa...")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    cph.baseline_survival_.plot(ax=ax2, color='crimson', linewidth=3, legend=False)
    ax2.set_title('Curva de Retenção Base da ArqDigital', fontsize=14, weight='bold')
    ax2.set_xlabel('Tempo (Meses de Casa)', fontsize=12)
    ax2.set_ylabel('Probabilidade de Retenção', fontsize=12)
    ax2.set_ylim([0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    caminho_fig2 = os.path.join(pasta_logs, 'Curva_Sobrevivencia_Base.png')
    fig2.savefig(caminho_fig2, bbox_inches='tight', dpi=300)
    plt.close(fig2)

    logger.info(f"SUCESSO: Gráficos de Sobrevivência salvos na pasta {pasta_logs}")

def run_training(df):
    """
    Inicio do treino completo
    :param df: 
    :return: 
    """
    logger.info("[INICIANDO]: Inicio do treino completo")

    # Limpeza e engenharia de Flags
    df_clean = drop_leakage_columns(df)

    # Mimitezando as flags campeãs
    df_clean['is_perfil_Nao_Mapeado'] = np.where(
        df_clean['perfil_comportamental'].str.contains('Não Mapeado', na = False), 1, 0
    )

    df_clean['is_dep_RELACIONAMENTO'] = np.where(
        df_clean['departamento_nome_api'] == 'RELACIONAMENTO', 1, 0
    )

    # Definição das FEATURES CAMPEÃS (ADICIONANDO A FAIXA SALARIAL)
    # Para o COX, precisamos da coluna de tempo e a de evento
    colunas_cox = [
        'meses_de_casa', # Relógio
        'target_pediu_demissao', # Evento
        'faixa_salarial', # Categórica
        'faixa_idade', # Categórica
        'is_com_dependentes', # Binária
        'is_perfil_Nao_Mapeado', # Binária
        'is_dep_RELACIONAMENTO' # Binária
    ]

    df_final = df_clean[colunas_cox].copy()

    # Tratamento categórico (O COX so entende números)
    # Realizando o dummies das features categoricas
    df_final = pd.get_dummies(df_final, columns=['faixa_salarial', 'faixa_idade'], drop_first=True)

    # Treinamento com a regularização (L2 Penalty)
    cph = CoxPHFitter(penalizer = 0.1)

    logger.info("[INICIANDO]: Inicializando o FIT do Cox Proportional Hazards")
    cph.fit(
        df_final,
        duration_col = 'meses_de_casa',
        event_col = 'target_pediu_demissao'
    )

    # Salvando toda a inteligência
    pasta_modelos = os.path.join(BASE_DIR, "Models")
    os.makedirs(pasta_modelos, exist_ok = True)
    caminho_modelo = os.path.join(pasta_modelos, "cox_turnover_model.pkl")
    joblib.dump(cph, caminho_modelo)

    # Visualziação final
    evaluate_survival_model(cph, df_final)

    logger.info(f"[FINALIZAÇÃO]: MODELO DE SOBREVIVÊNCIA SALVO: {caminho_modelo}")

    return cph