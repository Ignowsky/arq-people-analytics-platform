# SRC/DataExtractor.py

# Importando as libs
import os
import pandas as pd
from sqlalchemy.engine import Engine

# Importandos os módulos da pasta raiz.
try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

# iniciando o logger
logger = setup_logger(__name__)

class DataExtractor:
    """
    Classe responsável pela extração de dados do DW e gerenciamento dos backups locais bruto (Raw Data).
    """

    def __init__(self, engine: Engine, schema: str):
        """
        Inicializa o extrator recebendo a conexão já estabelecida.
        :param engine:  (Engine): Conexão ativa do SQLAlchemy.
        :param schema: Nome do schema para direcionamento das queries
        """

        self.engine = engine
        self.schema = schema

        # Mapeamento do diretório raiz do projeto
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def extract_obt_turnover(self) -> pd.DataFrame:
        """
        Executa a query de extração da One Big Table (OBT) de turnover.
        :param self:
        :return: pd.DataFrame contendo os dados brutos consolidados.
        """
        logger.info(f"[INICIANDO]: Extração da One Big Table (OBT) de turnover.")
        query = f'SELECT * FROM {self.schema}.vw_obt_turnover_lr;'

        try:
            df_raw = pd.read_sql(query, self.engine)
            logger.info(f"[SUCESSO]: Extração concluída com sucesso. Sahe dos Dados: {df_raw.shape}")
            return df_raw

        except Exception as e:
            logger.error(f"[ERROR]: Falha na execução da query no DW. Motivo {e}")
            raise


    def save_raw_backup(self, df: pd.DataFrame, file_name: str = "obt_turnover_bruta.csv") -> None:
        """
        Salva uma cópia de segurança dos dados brutos no diretório de backup (Data/Raw)
        :param self:
        :param df: Dataframe emitido pela query do DW
        :param file_name: nome do arquivo csv, que iremos salvar
        :return: None
        """

        try:
            pasta_raw = os.path.join(self.base_dir, "Data", "Raw")
            os.makedirs(pasta_raw, exist_ok=True)
            # Gravando o arquivo no diretorio
            caminho_completo = os.path.join(pasta_raw, file_name)
            df.to_csv(caminho_completo, index=False)

            logger.info(f"[SUCCESSO]: Arquivo {file_name} salvo com sucesso.")

        except Exception as e:
            logger.error(f"[ERROR]: Erro ao tentar salvar o arquivo no disco: {e}")
            raise