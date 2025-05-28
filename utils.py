# utils.py
import logging
import sys
import config # Para obtener el nivel de log desde config

def get_logger(name, level_str=None):
    """Configura y devuelve un logger."""
    if level_str is None:
        level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level_str, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers: # Evitar añadir múltiples handlers si se llama varias veces
        logger.setLevel(log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Handler para la consola
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # Opcional: Handler para archivo
        # fh = logging.FileHandler('trading_agent.log', mode='a') # 'a' para append
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)
            
    # Ajustar el nivel si ya existe el logger pero queremos un nivel diferente
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
        
    return logger

# Crear un logger global para usar en otros módulos si es necesario
# o cada módulo puede llamar a get_logger(__name__)
logger = get_logger('main_app') # Logger principal para la app