import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Importando os dados do Logger
try:
    from .logger import setup_logger
except ImportError:
    from logger import setup_logger

logger = setup_logger(__name__)


class DataConnection:
    """
    Classe responsável pela conexão com o banco de dados PostgreSQL.
    Realiza o mapeamento absoluto das credenciais de ambiente.
    """

    def __init__(self):
        # 1. Mapeamento Absoluto do arquivo .env (Força a leitura na raiz do projeto)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(BASE_DIR, '.env')

        # O parâmetro override=True garante que o arquivo local tenha prioridade
        load_dotenv(dotenv_path=env_path, override=True)

        # 2. Carregamento das credenciais
        self.credentials = {
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASS"),
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
            'dbname': os.getenv("DB_NAME"),
            'schema': os.getenv("DB_SCHEMA")
        }

    def connect_to_db(self):
        """
        Cria e retorna a engine de conexão com o banco de dados PostgreSQL usando SQLAlchemy.
        """
        # 3. Auditoria de Falha (Diagnóstico preciso do que está faltando)
        chaves_vazias = [k for k, v in self.credentials.items() if not v]

        if chaves_vazias:
            logger.error(
                f"[ERRO CRÍTICO] As seguintes credenciais não foram encontradas ou estão vazias no .env: {chaves_vazias}")
            sys.exit(1)

        url = f"postgresql://{self.credentials['user']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['dbname']}"

        try:
            # Limpeza: remove os espaços invisíveis do final do schema
            schema_limpo = self.credentials['schema'].strip()

            # Criação da Engine
            engine = create_engine(url, connect_args={
                "options": f"-c search_path={schema_limpo}"
            })

            logger.info(
                f"[SUCESSO] Conexão estabelecida com o banco '{self.credentials['dbname']}' (Schema: '{schema_limpo}').")

            return engine, schema_limpo

        except Exception as e:
            logger.error(f"[ERRO CRÍTICO] Falha de autenticação ao conectar no DW: {e}")
            sys.exit(1)