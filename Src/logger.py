import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logger(nome_modulo=None):
    """
    Configura e retorna um logger padronizado S-Rank.
    Salva logs na pasta raiz do projeto (Logs/) e mostra no terminal.
    """
    # 1. O Jutsu de Localização Absoluta: Descobre a raiz do projeto dinamicamente
    # __file__ é o logger.py. O dirname dele é 'Src'. O dirname do 'Src' é a raiz do projeto.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 2. Define o caminho absoluto para a pasta de logs usando os.path.join
    pasta_logs = os.path.join(BASE_DIR, 'Logs')
    os.makedirs(pasta_logs, exist_ok=True)  # Cria a pasta se não existir, blindado pro Linux

    # Adicionei a extensão .log pra ficar no padrão corporativo
    arquivo_log = os.path.join(pasta_logs, 'logs_lr_model.log')

    # 3. Configura o formatador (Data - Nível - Modulo - LINHA DO ERRO - Mensagem)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s:%(lineno)d - %(message)s')

    # 4. Handler de Arquivo (Rotativo: max 5MB, guarda 3 arquivos antigos)
    file_handler = RotatingFileHandler(
        filename=arquivo_log,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 5. Handler de Console (Terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 6. Obtém o logger e evita duplicação de handlers
    logger = logging.getLogger(nome_modulo if nome_modulo else __name__)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger