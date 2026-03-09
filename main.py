import sys
import os

# Jutsu de Localização Absoluta S-Rank: Ancorando na raiz do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Src"))

# Importando os Capitães de cada Batalhão (Facade Pattern S-Rank)
from Src.logger import setup_logger
from Src.data_extraction import extract_data_from_dw, save_raw_backup
from Src.data_cleaning import run_data_cleaning
from Src.feature_engineering import run_feature_engineering
from Src.train import run_training

logger = setup_logger("MAIN_ORCHESTRATOR")


def rodar_esteira_mlops():
    """
    O General supremo: Delega as tarefas na ordem exata da nossa arquitetura.
    """
    logger.info("🔥 INICIANDO O MEGAZORD: ESTEIRA DE MLOPS S-RANK 🔥")
    logger.info("=" * 60)

    try:
        # ---------------------------------------------------------
        # FASE 1: O GARIMPO (Extração do Banco de Dados)
        # ---------------------------------------------------------
        logger.info("▶️ FASE 1: Extração de Dados do PostgreSQL...")
        df_raw = extract_data_from_dw()
        save_raw_backup(df_raw)  # Salva o "Save State" na pasta Raw
        logger.info("✅ FASE 1 CONCLUÍDA.")
        logger.info("-" * 30)

        # ---------------------------------------------------------
        # FASE 2: A FAXINA (Limpeza e Padronização)
        # ---------------------------------------------------------
        logger.info("▶️ FASE 2: Limpeza de Nulos, Datas e Categorias...")
        # O run_data_cleaning vai ler o CSV da pasta Raw, limpar e salvar
        df_limpo = run_data_cleaning("obt_turnover_bruta.csv")
        logger.info("✅ FASE 2 CONCLUÍDA.")
        logger.info("-" * 30)

        # ---------------------------------------------------------
        # FASE 3: A FORJA (Engenharia de Features)
        # ---------------------------------------------------------
        logger.info("▶️ FASE 3: Forjando Variáveis Matemáticas (Idade, Tempo de Casa)...")
        # O run_feature_engineering recebe o df_limpo direto da memória RAM
        df_features = run_feature_engineering(df_limpo)

        # 🛡️ BLINDAGEM S-RANK: Garantindo caminhos no Linux do Render
        pasta_processed = os.path.join(BASE_DIR, "Data", "Processed")
        os.makedirs(pasta_processed, exist_ok=True)  # Cria a pasta se o Render tiver apagado
        caminho_processed = os.path.join(pasta_processed, "obt_turnover_preparada.csv")

        # Salvando o Checkpoint Final antes da matemática pura
        df_features.to_csv(caminho_processed, index=False)
        logger.info(f"Checkpoint S-Rank salvo em: {caminho_processed}")
        logger.info("✅ FASE 3 CONCLUÍDA.")
        logger.info("-" * 30)

        # ---------------------------------------------------------
        # FASE 4: O COMBATE (Machine Learning)
        # ---------------------------------------------------------
        logger.info("▶️ FASE 4: Treinamento da IA (Regressão Logística + SMOTE)...")
        # Passa a bola final pro treinamento que vai fatiar, aplicar o SMOTE e salvar no disco
        run_training(df_features)

        logger.info("=" * 60)
        logger.info("🚀 ESTEIRA FINALIZADA COM SUCESSO! O ALGORITMO ESTÁ VIVO E TREINADO.")


    except Exception as e:
        logger.error(f"❌ FALHA CATASTRÓFICA NA ESTEIRA: {e}")
        logger.error("O efeito dominó foi interrompido para evitar corrupção de dados.")


if __name__ == "__main__":
    rodar_esteira_mlops()