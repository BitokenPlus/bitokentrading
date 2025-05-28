# config.py (Versión para Depuración - Genérica Sin Key para Futuros)

# --- API Keys ---
# ¡¡IMPORTANTE!! Dejar VACÍAS para acceso público sin autenticación.
# config.py (Versión CORREGIDA con límites de datos aumentados)

# --- API Keys ---
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# --- Configuración General ---
MARKET_TYPE = 'future'
TARGET_QUOTE_ASSET = "USDT"
APP_VERSION = "AI-Cervello 🧠" # Incrementar versión

# --- Configuración de Obtención de Datos ---
# ¡¡AUMENTAR ESTOS VALORES!!
DATA_FETCH_LIMIT_LTF = 350  # Mínimo recomendado, 400-500 es mejor si la API lo permite sin ser muy lento
DATA_FETCH_LIMIT_HTF = 450  # Mínimo recomendado
LTF = '1h' # Temporalidad por defecto
HTF_DERIVATION_MAP = {'1m':'5m','5m':'15m','15m':'1h','30m':'2h','1h':'4h','2h':'6h','4h':'1d','6h':'1d','12h':'1d','1d':'1w'}
TICKER_CHUNK_SIZE = 75 # Reducido un poco de 100
GLOBAL_MARKET_SYMBOL = "BTC/USDT:USDT"

# --- Modos de Selección de Pares y Filtros ---
PAIR_SELECTION_MODE = "ADAPTIVE_FOCUS" # Cambiado a un modo más inteligente por defecto
SPECIFIC_PAIRS_TO_MONITOR = []

# Umbrales para los modos (Ajusta estos a valores iniciales razonables)
ADAPTIVE_MIN_VOL = 5_000_000       # 5M USD
ADAPTIVE_MIN_OI = 1_000_000         # 1M USD
ADAPTIVE_TOP_N = 30                 # Top 30 pares

STRICT_MIN_VOLUME = 50_000_000      # 50M USD
STRICT_MIN_OI = 10_000_000          # 10M USD
TOP_N_LIQUID = 20                   # Top 20 más líquidos

USER_MIN_VOLUME = 10_000_000        # Valor inicial para UI si se usa USER_DEFINED_FILTERS
USER_MIN_OI = 2_000_000             # Valor inicial para UI

TOP_N_BY_OI = 50                    # Usado como fallback o por USER_DEFINED_FILTERS

# Ajuste automático de filtros
FILTER_ADJUSTMENT_RETRIES = 2
FILTER_ADJUSTMENT_FACTOR = 0.6 # Reducir al 60%

# --- Parámetros de Indicadores ---
RSI_PERIOD = 14; RSI_OB = 70; RSI_OS = 30
MACD_F = 12; MACD_S = 26; MACD_SIG = 9
VOL_MA_P = 20
FIB_LOOKBACK = 150 # ¡ASEGÚRATE QUE DATA_FETCH_LIMIT_LTF sea mayor que esto + periodos de otros indicadores!
SMA_S_P = 20; SMA_L_P = 50
EMA_S_P = 12; EMA_L_P = 26 # EMA Larga también para tendencia HTF
BB_P = 20; BB_STD = 2.0
ADX_P = 14; ATR_P = 14
ICHIMOKU_TENKAN = 9; ICHIMOKU_KIJUN = 26; ICHIMOKU_SENKOU_B = 52

# --- Parámetros de Market Profile ---
VOLATILITY_LOOKBACK = 20 # Para ATR normalizado en MarketProfile
TREND_STRENGTH_LOOKBACK = 14 # Para ADX en MarketProfile
MARKET_REGIME_EMA_SHORT = 10
MARKET_REGIME_EMA_LONG = 30 # Usado para perfil de mercado

# --- Parámetros de Estrategia "Cervello" ---
CERVELLO_ADX_THRESHOLD = 23 # Umbral ADX para considerar tendencia fuerte
CERVELLO_VOL_SPIKE_FACTOR = 1.7
CERVELLO_OI_HIGH_THRESHOLD = 20_000_000 # OI "alto"
CERVELLO_SIGNAL_CONFIDENCE_THRESHOLD = 0.55 # Umbral de confianza
# Pesos para la estrategia (AJUSTA ESTOS CUIDADOSAMENTE)
CERVELLO_WEIGHTS = {
    'market_regime': 2.0,
    'htf_trend_strong': 1.8, 'htf_trend_weak': 0.7,
    'ltf_trend_strong': 1.5, 'ltf_trend_weak': 0.6,
    'rsi_optimal': 0.8, 'rsi_divergence': 1.5, # Divergencia tiene más peso
    'macd_crossover': 0.0, # Podrías no usarlo si ya usas la tendencia LTF completa
    'macd_momentum': 0.5,  # Para el histograma
    'volume_confirmation': 0.6,
    'bb_squeeze_breakout': 0.0, # Placeholder, necesita lógica de squeeze
    'bb_reversion_extreme': 0.8, # Si toca BBL/BBU y revierte
    'oi_confirmation': 0.4,      # OI aumentando con la tendencia
    'funding_rate_favorable': 0.2,
    'funding_rate_contrarian': -0.4, # Penalización si FR es muy adverso
    'fib_bounce_strong': 1.0,
    'ichimoku_strong_signal': 1.5
}
# Parámetros de TP/SL para CervelloStrategy
SL_ATR_MULTIPLIER = 1.5
TP1_ATR_MULTIPLIER = 1.0
TP2_ATR_MULTIPLIER = 2.0
TP3_ATR_MULTIPLIER = 3.0 # Reducido ligeramente
MIN_RISK_REWARD_RATIO = 1.5

# --- Logging ---
LOG_LEVEL = "DEBUG" # MANTENER EN DEBUG HASTA QUE TODO FUNCIONE