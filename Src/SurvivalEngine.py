import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, PrecisionRecallDisplay
from lifelines import CoxPHFitter

try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger(__name__)


class SurvivalEngine:
    """
    Motor de Machine Learning.
    Recebe os dados processados, treina os algoritmos de classificação (Logística)
    e de tempo de evento (Cox), avalia as métricas e salva os modelos em disco.
    """

    def __init__(self, penalizer: float = 0.1):
        """
        Inicializa o motor com os hiperparâmetros estabelecidos.
        """
        # Configuração do Pipeline de Classificação
        smote = SMOTE(random_state=42, k_neighbors=2)
        lr = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', C=0.1)
        self.lr_pipeline = Pipeline(steps=[
            ('smote', smote),
            ('classifier', lr)
        ])

        # Configuração do Modelo de Sobrevivência
        self.cph = CoxPHFitter(penalizer=penalizer)

        # Mapeamento de Diretórios Absolutos
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'Models')
        self.logs_path = os.path.join(self.base_dir, 'Logs')

        self.lr_path = os.path.join(self.models_dir, 'lr_turnover_model.pkl')
        self.cph_path = os.path.join(self.models_dir, 'cox_turnover_model.pkl')

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, duration_col: str = 'meses_de_casa',
                    event_col: str = 'target_pediu_demissao'):
        """
        Executa o ajuste (fit) dos modelos matemáticos em paralelo.
        Aplica o isolamento da variável de tempo para a Regressão Logística.
        """
        # --- 1. TREINAMENTO DA REGRESSÃO LOGÍSTICA ---
        logger.info("[SurvivalEngine] Iniciando a calibração da Regressão Logística...")

        # Remoção da variável temporal exclusiva para o classificador
        X_lr = X_train.drop(columns=[duration_col]).copy() if duration_col in X_train.columns else X_train.copy()
        self.lr_pipeline.fit(X_lr, y_train)

        # --- 2. TREINAMENTO DO MODELO COX PROPORTIONAL HAZARDS ---
        logger.info("[SurvivalEngine] Inicializando o ajuste do modelo Cox Proportional Hazards...")

        # Unificação do conjunto de dados (A biblioteca lifelines exige features e target no mesmo DataFrame)
        df_cox_train = X_train.copy()
        df_cox_train[event_col] = y_train.values

        self.cph.fit(
            df_cox_train,
            duration_col=duration_col,
            event_col=event_col
        )

        logger.info("[SurvivalEngine] Treinamento dos modelos concluído com sucesso.")
        return self.lr_pipeline, self.cph

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, duration_col: str = 'meses_de_casa'):
        """
        Gera os relatórios estatísticos de ambos os modelos.
        """
        logger.info("[SURVIVAL ENGINE]: Avaliando a precisão estatística da Regressão Logística...")
        os.makedirs(self.logs_path, exist_ok=True)

        # 1. Avaliação do Classificador (Limiar Ajustado de 0.39)
        X_lr_test = X_test.drop(columns=[duration_col]) if duration_col in X_test.columns else X_test
        probabilidades = self.lr_pipeline.predict_proba(X_lr_test)[:, 1]
        y_pred_adj = (probabilidades >= 0.39).astype(int)

        print('\n' + '=' * 60)
        print('--- RELATÓRIO DE MÉTRICAS (LOGÍSTICA - LIMIAR 0.39) ---')
        print('=' * 60)
        print(classification_report(y_test, y_pred_adj))

        self._plot_logistic_metrics(X_lr_test, y_test, y_pred_adj)

        # 2. Avaliação do Modelo de Sobrevivência
        logger.info("[SURVIVAL ENGINE]: Avaliando a estabilidade do Modelo Cox...")

        df_cox_test = X_test.copy()
        df_cox_test[y_test.name] = y_test.values

        try:
            # Calculando o C-index com dados inéditos
            c_index_test = self.cph.score(df_cox_test, scoring_method = "concordance_index")
            logger.info(f"Concordance Index (Cox - Teste Frio): {np.round(c_index_test, 2)}")
        except Exception as e:
            logger.warning(f"[WARNING] Ao calcular o C-Index no teste (pode ocorrer holdout se for muito pequeno): {e}")


        print('\n' + '=' * 60)
        print('--- RAIO-X DAS VARIÁVEIS (COX MODEL SUMMARY) ---')
        print('=' * 60)
        print(self.cph.print_summary())
        print('=' * 60)

        self._plot_survival_metrics()

    def _plot_logistic_metrics(self, X_test, y_test, y_pred):
        """Salva a matriz de confusão e a curva Recall-Precision."""
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16, "weight": "bold"}, ax=ax)
        ax.set_xticklabels(['Retido (0)', 'Evasão (1)'], fontsize=12)
        ax.set_yticklabels(['Retido (0)', 'Evasão (1)'], fontsize=12, rotation=0)
        ax.set_title('Matriz de Confusão - Logística', fontsize=14, weight='bold')
        fig.savefig(os.path.join(self.logs_path, 'Matriz_Confusao_LR.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
        PrecisionRecallDisplay.from_estimator(self.lr_pipeline, X_test, y_test, name="Logística", ax=ax_pr,
                                              color="crimson")
        ax_pr.set_title("Curva Recall-Precision", fontsize=14, weight="bold")
        fig_pr.savefig(os.path.join(self.logs_path, "Curva_Recall_Precision.png"), bbox_inches="tight", dpi=300)
        plt.close(fig_pr)

    def _plot_survival_metrics(self):
        """Salva os gráficos de impacto das variáveis e a curva base do Cox."""
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        self.cph.plot(ax=ax1)
        ax1.set_title('Impacto no Risco de Turnover (Hazard Ratios)', fontsize=14, weight='bold')
        plt.tight_layout()
        fig1.savefig(os.path.join(self.logs_path, 'Cox_Impacto_Variaveis.png'), bbox_inches='tight', dpi=300)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        self.cph.baseline_survival_.plot(ax=ax2, color='crimson', linewidth=3, legend=False)
        ax2.set_title('Curva de Retenção Base da ArqDigital', fontsize=14, weight='bold')
        ax2.set_xlabel('Tempo (Meses de Casa)', fontsize=12)
        ax2.set_ylabel('Probabilidade de Retenção', fontsize=12)
        ax2.set_ylim([0, 1.05])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        fig2.savefig(os.path.join(self.logs_path, 'Curva_Sobrevivencia_Base.png'), bbox_inches='tight', dpi=300)
        plt.close(fig2)

        logger.info(f"[SURVIVAL ENGINE]: Gráficos salvos com sucesso na pasta {self.logs_path}")

    def save_models(self):
        """Salva as inteligências em formato .pkl."""
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.lr_pipeline, self.lr_path)
        joblib.dump(self.cph, self.cph_path)
        logger.info("[SURVIVAL ENGINE]: Modelos exportados para a pasta Models com sucesso.")