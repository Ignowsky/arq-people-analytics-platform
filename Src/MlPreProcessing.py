import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import joblib # Pra salvar o nosso túnel de transformação

try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger(__name__)

class MlPreProcessingMixin:
    """
Bloco 3: Responsável pela divisão entre treino e teste do dataset
    e aplicação de transformações matemáticas (StandardScaler).
    Projetado para ser herdado pela classe DataProcessor.
    """

    def split_train_test(self, df: pd.DataFrame, target_name: str = 'target_pediu_demissao') -> tuple:
        """
        Fatia o dataset em Treino (pra aprender) e Teste (pra validar),
        garantindo a mesma proporção de turnover nos dois (stratify).

        :param df: dataframe
        :param target_name: target_pediu_demissao
        :return: X_train, X_test, y_train, y_test
        """

        logger.info(f"[ML PREPROC]: Iniciando o fatiamento entre treino e teste (80/20)")
        df_processed = df.copy()

        X = df_processed.drop(columns = target_name)
        y = df_processed[target_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

        logger.info(f"[DIVISÃO TREINO]: X = {X_train.shape}, y = {y_train.shape}")
        logger.info(f"[DIVISÃO TESTE]: X = {X_test.shape}, y = {y_test.shape}")

        return X_train, X_test, y_train, y_test

    def apply_scaler(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Aplica o StandardScaler garantindo a preservação integral das colunas.
        """
        logger.info(f"[ML PREPROC] Colunas recebidas para escalonamento: {list(X_train.columns)}")

        # 1. Definir o que deve ser escalonado (apenas colunas contínuas existentes)
        num_features = [col for col in ['meses_de_casa', 'salario_contratual', 'idade', 'qtd_dependentes'] if
                        col in X_train.columns]

        # 2. Definir o que deve passar direto (Dummies, Flags e o Relógio se ele não for escalonado)
        # Se você quer o meses_de_casa SEM escala para o Cox, tire ele da lista num_features
        remainder_cols = [col for col in X_train.columns if col not in num_features]

        # 3. Configurar o Transformer com Passthrough obrigatório
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), num_features)
            ],
            remainder='passthrough'
        )

        # 4. Executar a transformação (Retorna NumPy Array)
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)

        # 5. RECONSTRUÇÃO CRÍTICA: O ColumnTransformer empilha [num_features + remainder_cols]
        colunas_finais = num_features + remainder_cols

        X_train_final = pd.DataFrame(X_train_scaled, columns=colunas_finais, index=X_train.index)
        X_test_final = pd.DataFrame(X_test_scaled, columns=colunas_finais, index=X_test.index)

        # 6. Forçar casting para garantir que o Cox não rejeite os dados
        X_train_final = X_train_final.astype(float)
        X_test_final = X_test_final.astype(float)

        logger.info(f"[ML PREPROC] DataFrame reconstruído com as colunas: {list(X_train_final.columns)}")

        return X_train_final, X_test_final