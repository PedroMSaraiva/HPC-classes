"""
Módulo para configuração do logger
"""
from loguru import logger
import sys
from pathlib import Path

def setup_logger():
    """
    Configura o logger com formatação avançada para console e arquivo.
    
    Returns:
        logger: Objeto logger configurado
    """
    # Remove a configuração padrão
    logger.remove()
    
    # Adiciona handler para o console com cores e formatação avançada
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        level="INFO"
    )
    
    # Cria diretório de logs se não existir
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Adiciona handler para arquivo de log com rotação
    logger.add(
        "logs/random_forest_{time}.log",
        rotation="100 MB",
        retention="1 week",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    return logger 