# SRC/feature_engineering
# Importanções iniciais

import numpy as np
import pandas as pd
import datetime as date

# Importando os dados do Logger.py
from .logger import setup_logger
logger = setup_logger(__name__)

# Criação de novas colunas temporarias
def creating_cutoff_date(df):
    """
    Função com a inteção de criar um novo dataset com a data corte

    :param df: dataframe
    :return: dataframe new column
    """
    logger.info("INICIANDO: Criação de coluna temporaria")

    df_clean = df.copy()

    df_clean['data_corte'] = df_clean['data_demissao'].fillna(pd.Timestamp.today())

    return df_clean

# Criação das colunas de idade e tempo de casa em meses

def creating_age_column(df):
    """
    Criação da coluna de idade com base na data de corte e na data de nascimento

    :param df: dataframe
    :return: dataframe new column
    """
    logger.info("INICIANDO: Criação da coluna de idade")
    df_clean = df.copy()

    df_clean['idade'] = ((df_clean['data_corte'] - df_clean['data_nascimento']).dt.days // 365).astype(int)

    return df_clean

def creating_hometime_column(df):
    """
    Criação da coluna de tempo de casa com base na data de corte e na data de admissao

    :param df: dataframe
    :return: dataframe new column
    """
    logger.info("INICIANDO: Criação da coluna de tempo de casa em meses")
    df_clean = df.copy()

    df_clean['meses_de_casa'] = ((df_clean['data_corte'] - df_clean['data_admissao']).dt.days / 30).astype(float).round(2)

    return df_clean

def creating_region_column(df):
    """
    Criação da coluna de região com base no cep que possuimos no dw

    :param df: dataframe
    :return: dataframe new column
    """

    df_clean = df.copy()

    df_clean['zona_cep'] = df['cep'].astype(str).str.zfill(8).str[:2]

    df_clean = df_clean.drop(columns = ['cep'])

    return df_clean

def creating_salary_range(df):
    """
    Cria categorias salariais usando a função pd.qcut do pandas, que seapra o dataset em 3 faixas perfeitamentes iguais baseada na distribuição real da base
    :param df: df_bruto
    :return: df_clean
    """

    logger.info("[INICIANDO]: Criação de uma nova feature de faixas salariais")
    df_clean = df.copy()

    # Utilizando o qcut que divide em 3 faixas balanceadas
    df_clean['faixa_salarial'] = pd.qcut(
        df_clean['salario_contratual'],
        q = 3,
        labels = ['Baixo', 'Médio', 'Alto']
    )

    df_clean = df_clean.drop(columns = ['salario_contratual'])

    return df_clean

def creating_demographinc_ranges(df):
    logger.info("[INICIANDO]: Binarização de Dependentes e Fatiamento de Idades")

    df_clean = df.copy()

    # Regra do zero: Tem ou não dependentes? (1 ou 0)

    df_clean['is_com_dependentes'] = np.where(
        df_clean['qtd_dependentes'] > 0, 1, 0
    )

    # Fatiamento Geracional e momento de carreira
    # Cortes definidos: até 25 anos | de 26 a 35 anos | acima de 35 anos
    bins_idade = [0, 25, 35, 100]
    labels_idade = ['Ate_25', '26_a_35', 'Acima_35']

    df_clean['faixa_idade'] = pd.cut(
        df_clean['idade'],
        bins = bins_idade,
        labels = labels_idade
    )

    # Limpando o dataset para evitar multicolinearidade
    df_clean = df_clean.drop(columns = ['idade', 'qtd_dependentes'])

    return df_clean

def run_feature_engineering(df):
    """
    Capitão da Forja: Orquestra a criação de todas as variáveis matemáticas.
    Recebe o DataFrame limpo e devolve ele anabolizado com as novas features.
    """
    logger.info("Iniciando a orquestração do Feature Engineering...")

    try:
        # O pipeline de criação (Efeito dominó)
        df_feat = creating_cutoff_date(df)
        df_feat = creating_age_column(df_feat)
        df_feat = creating_hometime_column(df_feat)
        df_feat = creating_salary_range(df_feat)
        df_feat = creating_demographinc_ranges(df_feat)
        df_feat = creating_region_column(df_feat)

        # Opcional: se você modularizou aqueles agrupamentos (DRY)
        # df_feat = group_infrequent_categories(df_feat, 'departamento_nome_api')

        logger.info(f"SUCESSO: Forja de features concluída. Shape atual: {df_feat.shape}")
        return df_feat

    except Exception as e:
        logger.error(f"ERROR: Falha catastrófica na forja de features: {e}")
        raise