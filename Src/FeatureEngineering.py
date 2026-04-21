# Src/FeatureEngineering.py
import pandas as pd
import numpy as np
import datetime as date
# Importando o logger de forma blindada
try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineeringMixin:
    """
    Bloco responsável por alicar todas as regras de negócios definidas previamente pelo rh
    Projetado para ser herdado pela classe DataProcessor
    """

    def apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[FEATURE ENGINEERING]: Aplicando lógicas de negócio e criação de flags binárias...")
        df_processed = df.copy()
        # Flags Binárias Estratégicas
        df_processed['is_perfil_Nao_Mapeado'] = np.where(
            df_processed['perfil_comportamental'].str.contains('Não Mapeado', na=False), 1, 0
        )
        df_processed['is_dep_RELACIONAMENTO'] = np.where(
            df_processed['departamento_nome_api'] == 'RELACIONAMENTO', 1, 0
        )
        df_processed['is_com_dependentes'] = np.where(
            df_processed['qtd_dependentes'] > 0, 1, 0
        )

        return df_processed


    def creating_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Função com a inteção de criar um novo dataset com a data corte

        :param df: dataframe
        :return: dataframe new column
        """
        df_processed = df.copy()

        logger.info("[INICIANDO] Criação da coluna de Cutoff")
        df_processed['data_corte'] = df_processed['data_demissao'].fillna(pd.Timestamp.today())
        logger.info("[FINALIZANDO] Coluna de Cutoff criada com sucesso")

        logger.info("[INICIANDO] Criação da coluna de idade")
        df_processed['idade'] = ((df_processed['data_corte'] - df_processed['data_nascimento']).dt.days // 365).astype(int)
        logger.info("[FINALIZANDO] Coluna de idade criada com sucesso")

        logger.info("[INICIANDO] Criação da coluna de tempo de casa")
        df_processed['meses_de_casa'] = ((df_processed['data_corte'] - df_processed['data_admissao']).dt.days / 30.44).astype(float).round(2)
        logger.info("[FINALIZANDO] Coluna de tempo de casa criada com sucesso")

        return df_processed

    def discretize_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("[FEATURE ENGINEERING]: Discretizando variáveis contínuas (Idade e Salario)...")
        df_processed = df.copy()

        # Fatiamento salarial
        if 'salario_contratual' in df_processed.columns:
            df_processed['faixa_salarial'] = pd.qcut(
                df_processed['salario_contratual'], q = 3,
                labels = ['Baixo', 'Médio', 'Alto']
            )

        # Fatiamento de Idades
        if 'idade' in df_processed.columns:
            bins_idade = [0, 25, 35, 100]
            labels_idade = ['Ate_25', "26_a_35", 'Acima_35']
            df_processed['faixa_idade'] = pd.cut(
                df_processed['idade'], bins_idade, labels = labels_idade
            )

        return df_processed

    def run_feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orquestrador da ordem de execução de feature engineering
        :param df:
        :return:
        """

        logger.info(f"[INICIANDO] PIPELINE DE FEATURE ENGINEERING")
        df_processed = df.copy()

        # 1° Aplica as regras de negócios
        df_processed = self.apply_business_rules(df_processed)
        # 2° Cria as colunas de tempo e idade
        df_processed = self.creating_dates(df_processed)
        # 3° Discretiza oque foi criado acima
        df_processed = self.discretize_variables(df_processed)

        logger.info(f"[CONCLUÍDO] PIPELINE DE FEATURE ENGINEERING")
        return df_processed