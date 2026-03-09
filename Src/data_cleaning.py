import os
import pandas as pd
import numpy as np

# Importando o logger de forma blindada para o FastAPI e para rodar local
try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger(__name__)


def fill_categorical_nulls(df):
    logger.info(f"INICIANDO: Preenchimento de nulos (Missing Values)...")
    df_clean = df.copy()

    regras_preenchimento = {
        'cidade': 'Não Informado',
        'estado': 'Não Informado',
        'estado_civil': 'Não Informado',
        'nivel_hierarquico': 'Não Informado',
        'turno_trabalho': 'Não Informado',
        'tipo_contrato': 'Não Informado',
        'perfil_comportamental': 'Não Mapeado'
    }

    df_clean = df_clean.fillna(regras_preenchimento)

    # Correção: Adicionado o segundo .sum() pra retornar um número inteiro
    nulos_apos_limpeza = df_clean[list(regras_preenchimento.keys())].isna().sum().sum()
    logger.info(f"SUCESSO: Limpeza de categóricos. Nulos restantes nessas colunas: {nulos_apos_limpeza}")

    return df_clean


def cleaning_date_type(df, columns):
    logger.info("INICIANDO: Conversão de tipos de data...")
    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], format='%Y-%m-%d', dayfirst=True, errors='coerce')

    return df_clean


def group_infrequent_categories(df, column, threshold=5):
    logger.info(f"INICIANDO: Agrupamento da coluna '{column}' (Threshold: > {threshold})")
    df_clean = df.copy()

    if column in df_clean.columns:
        freq = df_clean[column].value_counts()
        categorias_manter = freq[freq > threshold].index
        df_clean[column] = np.where(
            df_clean[column].isin(categorias_manter),
            df_clean[column],
            'OUTROS'
        )
    return df_clean


def map_education(df, column='escolaridade'):
    logger.info("INICIANDO: Normalizando a coluna de escolaridade")
    df_clean = df.copy()

    dicionario_escolaridade = {
        'MBA': 'Pós-Graduação/MBA',
        'Pós Graduação': 'Pós-Graduação/MBA',
        'Pós Graduação (cursando)': 'Superior Completo',
        'Tecnólogo': 'Superior Completo',
        'Superior Completo': 'Superior Completo',
        'Superior (cursando)': 'Superior Incompleto/Cursando',
        'Superior Incompleto': 'Superior Incompleto/Cursando',
        'Médio Completo': 'Até Ensino Médio',
        'Fundamental Completo': 'Até Ensino Médio'
    }

    if column in df_clean.columns:
        df_clean[column] = df_clean[column].replace(dicionario_escolaridade)

    return df_clean


def run_data_cleaning(file_name="obt_turnover_bruta.csv"):
    logger.info("INICIANDO: Orquestração de limpeza de dados (Data Cleaning)...")

    # O Jutsu de Localização Absoluta S-Rank
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pasta_raw = os.path.join(BASE_DIR, "Data", "Raw")
    pasta_processed = os.path.join(BASE_DIR, "Data", "Processed")

    # Garante a existência da pasta no servidor Linux
    os.makedirs(pasta_processed, exist_ok=True)

    caminho_entrada = os.path.join(pasta_raw, file_name)
    caminho_saida = os.path.join(pasta_processed, "obt_turnover_limpo.csv")

    try:
        df = pd.read_csv(caminho_entrada)
        logger.info(f"SUCESSO: Base bruta carregada! Shape Inicial: {df.shape}")

        # 1. Datas
        colunas_de_data = ['data_nascimento', 'data_admissao', 'data_demissao']
        df = cleaning_date_type(df, colunas_de_data)

        # 2. Nulos
        df = fill_categorical_nulls(df)

        # 3. Agrupamentos (Trazendo seus soldados pro jogo)
        df = group_infrequent_categories(df, 'departamento_nome_api', threshold=5)
        df = group_infrequent_categories(df, 'perfil_comportamental', threshold=5)

        # 4. Padronização de Texto
        df = map_education(df, 'escolaridade')

        # Salvando o novo dataset limpo
        df.to_csv(caminho_saida, index=False)
        logger.info(f"SUCESSO: Base Limpa salva em: {caminho_saida}")

        return df

    except Exception as e:
        logger.error(f"ERROR: Falha crítica na lógica de limpeza {e}")
        raise