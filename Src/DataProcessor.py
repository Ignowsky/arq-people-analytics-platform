import pandas as pd

# Importando os blocos de mixins
from .DataCleaning import DataCleaningMixin
from .FeatureEngineering import FeatureEngineeringMixin
from .MlPreProcessing import MlPreProcessingMixin


try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger(__name__)

class DataProcessor(DataCleaningMixin, FeatureEngineeringMixin, MlPreProcessingMixin):
    """
    Classe central de Processamento de dados.
    Utiliza herença múltipla (Mixins) para agregar limpeza, engenharia de features,
    e pré-processamento de ML
    """

    def __init__(self):
        """
        Inicializa as variáveis globais que os Mixins vão consumir.
        """

        # Lista consumida pelo método drop_leakage() herdado do DataCleaning
        self.leakage_columns = [
            'colaborador_sk', 'data_nascimento', 'data_admissao',
            'data_demissao', 'data_corte'
        ]

        self.preprocessor = None # Guarda o motor do StandardScaler

        logger.info("[DATA PROCESSOR]: Instaciado com sucesso. Todos os módulos acoplados.")

    def run_full_pipeline(self, df: pd.DataFrame, target_name: str = 'target_pediu_demissao') -> tuple:
        # 1. Limpeza (Preservando as datas agora)
        df_limpo = self.run_cleaning_pipeline(df)

        # 2. Engenharia (Agora ele vai encontrar as datas para calcular meses_de_casa e faixa_idade)
        df_features = self.run_feature_engineering_pipeline(df_limpo)

        # 3. Agora sim, deletamos o que sobrou das datas brutas (Leakage)
        # Chamamos o método herdado do DataCleaning aqui no final
        df_features = self.drop_leakage(df_features)

        # ---------------------------------------------------------
        # SELEÇÃO DE FEATURES PARA O MODELO
        # ---------------------------------------------------------
        colunas_campeas = [
            'meses_de_casa',
            'faixa_salarial',
            'faixa_idade',
            'is_com_dependentes',
            'is_perfil_Nao_Mapeado',
            'is_dep_RELACIONAMENTO',
            target_name
        ]

        # Filtra e garante a existência
        df_model = df_features[[c for c in colunas_campeas if c in df_features.columns]].copy()

        # Auditoria para confirmar se as colunas sobreviveram
        print(f"\n[CHECKPOINT FINAL] Colunas prontas: {df_model.columns.tolist()}")

        # Encoding e Fatiamento
        cols_to_encode = [c for c in ['faixa_salarial', 'faixa_idade'] if c in df_model.columns]
        df_model = pd.get_dummies(df_model, columns=cols_to_encode, drop_first=True)

        X_train, X_test, y_train, y_test = self.split_train_test(df_model, target_name)
        X_train_scaled, X_test_scaled = self.apply_scaler(X_train, X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test