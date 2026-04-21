import sys
import os

# Configuração de caminhos absolutos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Src"))

# Importação de módulos
try:
    from Src.logger import setup_logger
    from Src.DataConnection import DataConnection
    from Src.DataExtractor import DataExtractor
    from Src.DataProcessor import DataProcessor
    from Src.SurvivalEngine import SurvivalEngine
except ImportError as e:
    print(f"[ERRO CRÍTICO] Falha na importação de módulos. Detalhes: {e}")
    sys.exit(1)

logger = setup_logger("MAIN_ORCHESTRATOR")


def rodar_esteira_mlops():
    """
    Orquestrador principal do pipeline de Machine Learning.
    Instancia as classes e executa as fases de processamento sequencialmente.
    """
    logger.info("=== INICIANDO PIPELINE DE MLOPS (TREINAMENTO) ===")

    try:
        # ---------------------------------------------------------
        # FASE 1: Conexão e Extração
        # ---------------------------------------------------------
        logger.info("[FASE 1] Iniciando conexão e extração de dados do PostgreSQL...")

        conexao = DataConnection()
        engine, schema = conexao.connect_to_db()

        extrator = DataExtractor(engine, schema)
        df_bruto = extrator.extract_obt_turnover()

        extrator.save_raw_backup(df_bruto)
        logger.info("[FASE 1] Extração e backup bruto concluídos com sucesso.")

        # ---------------------------------------------------------
        # FASE 2 e 3: Processamento de Dados
        # ---------------------------------------------------------
        logger.info("[FASE 2 e 3] Iniciando limpeza e engenharia de features...")

        processador = DataProcessor()
        X_train, X_test, y_train, y_test = processador.run_full_pipeline(df_bruto)

        logger.info("[FASE 2 e 3] Processamento concluído. Matrizes matemáticas preparadas.")

        # ---------------------------------------------------------
        # FASE 4: Treinamento dos Modelos
        # ---------------------------------------------------------
        logger.info("[FASE 4] Iniciando calibração dos modelos (Logistic Regression e Cox)...")

        motor_ai = SurvivalEngine(penalizer=0.1)

        motor_ai.train_model(
            X_train=X_train,
            y_train=y_train,
            duration_col = "meses_de_casa",
            event_col = "target_pediu_demissao"
        )

        motor_ai.evaluate_model(
            X_test=X_test,
            y_test=y_test,
            duration_col = "meses_de_casa"
        )

        motor_ai.save_models()
        logger.info("=== PIPELINE FINALIZADO COM SUCESSO ===")

    except Exception as e:
        logger.error(f"[ERRO CRÍTICO] Falha na execução da esteira: {e}")
        logger.error("Processo interrompido para evitar inconsistência de dados.")


if __name__ == "__main__":
    rodar_esteira_mlops()