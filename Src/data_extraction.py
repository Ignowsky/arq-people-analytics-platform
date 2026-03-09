# SRC/data_extraction.py

# Importando as libs
import os
import pandas as pd

# Importandos os módulos da pasta raiz.
try:
    from .logger import setup_logger
    from .database import connect_to_db
except ImportError:
    from logger import setup_logger
    from database import connect_to_db

# Iniciando o logger
logger = setup_logger(__name__)

# Criação da função de extração do DW
def extract_data_from_dw():
    """
    Função: Apenas conectar ao DW consolidado do time de gestão de pessoas e extrair a OBT para a memória em formato Dataframe
    :return: Dataframe pandas
    """
    logger.info("INICIALIZAÇÃO: Iniciando a extração da OBT de Turnover do DW")
    engine, schema = connect_to_db()

    try:
        query = f'SELECT * FROM "{schema}".vw_obt_turnover_lr;'
        df_raw = pd.read_sql(query, engine)
        logger.info(f"SUCESSO: Extração concluída com sucesso! shape dos dados {df_raw.shape}")
        return df_raw

    except Exception as e:
        logger.error(f"ERROR: Falha na comunicação com o DW verificar as conexão. Motivo {e}")
        raise

def save_raw_backup(df, file_name="obt_turnover_bruta.csv"):
    """
    Função: receber o dataframe gerado pela query e salvar na pasta Data/Raw
    :param df: dataframe
    :param file_name: nome fixo
    :return: csv salvo na pasta Raw
    """
    try:
        # O Jutsu de Localização Absoluta S-Rank
        # __file__ é o data_extraction.py. O dirname dele é 'Src'. O dirname do 'Src' é a raiz.
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Montando o caminho exato e garantindo a existência da pasta no Linux/Windows
        pasta_raw = os.path.join(BASE_DIR, "Data", "Raw")
        os.makedirs(pasta_raw, exist_ok=True)

        # Gravando o arquivo no disco
        caminho_completo = os.path.join(pasta_raw, file_name)
        df.to_csv(caminho_completo, index=False)

        logger.info(f"SUCESSO: Backup estático salvo em: {caminho_completo}")
    except Exception as e:
        logger.error(f"ERROR: Erro ao tentar salvar o arquivo no disco: {e}")
        raise


# ==========================================
# ÁREA DE TESTE (Orquestrando as funções isoladas)
# ==========================================
if __name__ == "__main__":
    # 1. Busca no banco
    df_obt = extract_data_from_dw()

    # 2. Salva o backup
    if not df_obt.empty:
        save_raw_backup(df_obt)
        print(df_obt.head())