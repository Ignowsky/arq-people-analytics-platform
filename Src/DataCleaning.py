# Src/DataCleaning.py

import os
import pandas as pd
import numpy as np

# Importando o logger de forma blindada para o FastAPI e para rodar local
try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger(__name__)


class DataCleaningMixin:
    """
    Bloco 1: Responsável pelo higienização bruta e estruturação de tipos.
    Projetado para ser herdado pela classe dataProcessor.
    """
    def fill_categorical_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[CLEANING]: Preenchimento de nulos (Missing Values)...")
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
        logger.info(f"[CLEANING]: Nulos restantes nas colunas mapeadas: {nulos_apos_limpeza}")


        return df_clean

    def cleaning_date_type(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        logger.info(f"[CLEANING]: Tratamento dos tipos de data")
        df_clean = df.copy()

        for col in columns:
            if col in df.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], format='%Y-%m-%d', errors='coerce')

        return df_clean


    def group_infrequent_categories(self, df: pd.DataFrame, column: str, threshold: int = 5) -> pd.DataFrame:
        logger.info(f"[CLEANING]: Agrupamento da coluna '{column}' (Threshold: > {threshold})...")
        df_clean = df.copy()

        if column in df_clean.columns:
            freq = df_clean[column].value_counts()
            categorias_manter = freq[freq >  threshold].index
            df_clean[column] = np.where(
                df_clean[column].isin(categorias_manter),
                df_clean[column],
                "OUTROS"
            )

        return df_clean

    def map_education(self, df: pd.DataFrame, column: str = 'escolaridade') -> pd.DataFrame:
        logger.info(f"[CLEANING]: Normalizando a coluna de escolaridade...")
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

    def drop_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas que o modelo pode não enxergar

        :param df:
        :return:
        """
        logger.info(f"[CLEANING]: Removendo colunas de Data Leakage (IDs e Data Soltas)...")
        df_clean = df.copy()

        colunas_presentes = [col for col in self.leakage_columns if col in df_clean.columns]
        df_clean = df_clean.drop(columns=colunas_presentes)

        return df_clean

    def run_cleaning_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("[INICIO]: PIPELINE DE DATA CLEANING INICIADO")
        df_clean = df.copy()

        colunas_de_data = ['data_nascimento', 'data_admissao', 'data_demissao']
        df_clean = self.cleaning_date_type(df_clean, colunas_de_data)
        df_clean = self.fill_categorical_nulls(df_clean)

        # REMOVA A LINHA ABAIXO:
        # df_clean = self.drop_leakage(df_clean)

        logger.info("[CLEANING]: DATA CLEANING FINALIZADO (DATAS PRESERVADAS)")
        return df_clean