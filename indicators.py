# indicators.py
import pandas as pd
import pandas_ta as ta
import numpy as np
from utils import get_logger # Asegúrate que utils.py está y define logger correctamente
import config # Para acceder a los periodos de los indicadores desde config.py
import math # Para math.isnan si es necesario en _calculate_fibonacci_retracement

logger = get_logger(__name__) # Logger específico para este módulo

class IndicatorCalculator:
    def __init__(self):
        logger.info("IndicatorCalculator (Corregido) inicializado.")
        # Cargar los períodos desde config.py para asegurar consistencia y nombres de columna
        self.rsi_period = getattr(config, 'RSI_PERIOD', 14)
        self.macd_fast = getattr(config, 'MACD_F', 12) # Usar MACD_F, MACD_S, MACD_SIG
        self.macd_slow = getattr(config, 'MACD_S', 26)
        self.macd_signal = getattr(config, 'MACD_SIG', 9)
        self.volume_ma_period = getattr(config, 'VOL_MA_P', 20) # Usar VOL_MA_P
        self.fib_lookback_period = getattr(config, 'FIB_LOOKBACK', 100)
        self.sma_short_period = getattr(config, 'SMA_S_P', 20)
        self.sma_long_period = getattr(config, 'SMA_L_P', 50)
        self.ema_short_period = getattr(config, 'EMA_S_P', 12)
        self.ema_long_period = getattr(config, 'EMA_L_P', 26)
        self.bb_period = getattr(config, 'BB_P', 20)
        self.bb_std_dev = float(getattr(config, 'BB_STD', 2.0)) # Asegurar que es float
        self.adx_period = getattr(config, 'ADX_P', 14) # Usar ADX_P
        self.atr_period = getattr(config, 'ATR_P', 14) # Usar ATR_P
        self.ichimoku_t = getattr(config, 'ICHIMOKU_TENKAN', 9)
        self.ichimoku_k = getattr(config, 'ICHIMOKU_KIJUN', 26)
        self.ichimoku_sb = getattr(config, 'ICHIMOKU_SENKOU_B', 52)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade un conjunto completo de indicadores técnicos al DataFrame.
        Todas las operaciones de pandas-ta se hacen sobre df_copy.
        """
        if df is None or df.empty:
            logger.warning("DataFrame vacío o nulo pasado a add_indicators.")
            return df 

        # Determinar longitud mínima de datos requerida por el indicador más largo
        # Esta es una estimación, algunos indicadores podrían necesitar un poco más para estabilizarse.
        min_len_required = max(
            self.rsi_period, self.macd_slow + self.macd_signal, self.sma_long_period, 
            self.ema_long_period, self.bb_period, self.adx_period + self.adx_period, # ADX necesita más (DI + suavizado)
            self.atr_period, self.ichimoku_sb, self.fib_lookback_period, self.volume_ma_period
        ) + 20 # Buffer adicional

        if len(df) < min_len_required:
            logger.warning(f"DataFrame para indicadores con datos insuficientes ({len(df)} velas, necesita ~{min_len_required}). "
                           "Algunos indicadores podrían ser NaN o faltar. Continuando cálculo...")
            # No retornamos aquí, intentamos calcular lo que se pueda. La estrategia debe manejar NaNs.

        df_copy = df.copy() # <<< --- IMPORTANTE: Crear la copia ANTES del bloque try ---

        try:
            # RSI
            df_copy.ta.rsi(length=self.rsi_period, append=True, 
                           col_names=(f'RSI_{self.rsi_period}',))

            # MACD
            df_copy.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True,
                            col_names=(f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
                                       f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}',
                                       f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'))
            
            # Medias Móviles Simples (SMA)
            df_copy.ta.sma(length=self.sma_short_period, append=True, col_names=(f'SMA_{self.sma_short_period}',))
            df_copy.ta.sma(length=self.sma_long_period, append=True, col_names=(f'SMA_{self.sma_long_period}',))

            # Medias Móviles Exponenciales (EMA)
            df_copy.ta.ema(length=self.ema_short_period, append=True, col_names=(f'EMA_{self.ema_short_period}',))
            df_copy.ta.ema(length=self.ema_long_period, append=True, col_names=(f'EMA_{self.ema_long_period}',))

            # Bandas de Bollinger
            bb_std_name_part = str(self.bb_std_dev).replace('.', '_') # Ej: 2.0 -> 2_0, o 2 -> 2
            df_copy.ta.bbands(length=self.bb_period, std=self.bb_std_dev, append=True,
                              col_names=(f'BBL_{self.bb_period}_{bb_std_name_part}',
                                         f'BBM_{self.bb_period}_{bb_std_name_part}',
                                         f'BBU_{self.bb_period}_{bb_std_name_part}',
                                         f'BBB_{self.bb_period}_{bb_std_name_part}',
                                         f'BBP_{self.bb_period}_{bb_std_name_part}'))
            
            # Volumen MA
            vol_ma_col_name = f'volume_ma_{self.volume_ma_period}'
            if 'volume' in df_copy.columns and df_copy['volume'].nunique() > 1 and not df_copy['volume'].isnull().all():
                df_copy.ta.sma(close=df_copy['volume'], length=self.volume_ma_period, append=True, 
                               col_names=(vol_ma_col_name,))
            else:
                logger.debug(f"Columna 'volume' no encontrada, es constante, o nula en df_copy. Añadiendo '{vol_ma_col_name}' como NaN.")
                df_copy[vol_ma_col_name] = np.nan # Añadir columna NaN para evitar KeyErrors

            # ATR (Average True Range)
            df_copy.ta.atr(length=self.atr_period, append=True, col_names=(f'ATR_{self.atr_period}',))

            # ADX (Average Directional Index)
            df_copy.ta.adx(length=self.adx_period, append=True, 
                           col_names=(f'ADX_{self.adx_period}', 
                                      f'DMP_{self.adx_period}', # +DI
                                      f'DMN_{self.adx_period}')) # -DI

            # Ichimoku Cloud
            df_copy.ta.ichimoku(tenkan=self.ichimoku_t, kijun=self.ichimoku_k, senkou_b=self.ichimoku_sb, 
                                append=True, 
                                col_names=(f'ISA_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}', 
                                           f'ISB_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}', 
                                           f'ITS_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}', 
                                           f'IKS_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}', 
                                           f'ICS_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}'))
            
            # Fibonacci Retracement Levels
            # Asegurarse que df_copy tenga suficientes datos ANTES de llamar a fibonacci
            if len(df_copy) >= self.fib_lookback_period:
                fib_levels = self._calculate_fibonacci_retracement(df_copy, lookback=self.fib_lookback_period)
                if fib_levels:
                    for level_pct, price in fib_levels.items(): # level_pct es 0.0, 23.6, etc.
                        df_copy[f'fib_{level_pct:.1f}'] = price 
            else:
                logger.warning(f"Datos insuficientes ({len(df_copy)}) para Fibonacci con lookback {self.fib_lookback_period}. No se calcularán Fibs.")


            logger.debug(f"Indicadores calculados para df_copy. Columnas finales: {list(df_copy.columns)}")
            return df_copy # Devolver la copia con los indicadores añadidos

        except Exception as e:
            logger.error(f"Error severo añadiendo indicadores: {e}", exc_info=True)
            # En caso de un error inesperado, es más seguro devolver el DataFrame original
            # para evitar que datos corruptos se propaguen, o None para forzar un error.
            # Devolver df_copy aquí podría tener algunos indicadores y otros no, lo cual es difícil de manejar.
            return df # Devolver el DataFrame original sin modificar si algo sale muy mal.

    def _calculate_fibonacci_retracement(self, df: pd.DataFrame, lookback: int) -> dict:
        """
        Calcula los niveles de retroceso de Fibonacci basados en el último swing high/low.
        Devuelve un diccionario con niveles (0.0, 23.6, ..., 100.0) y sus precios.
        0.0 es el inicio del swing (bajo en uptrend, alto en downtrend del swing).
        100.0 es el final del swing (alto en uptrend, bajo en downtrend del swing).
        """
        # Esta función no usa self.params directamente, usa el 'lookback' pasado.
        if df is None or len(df) < lookback or lookback < 2: # Necesita al menos 2 puntos para un swing
            logger.debug(f"Fibo: Datos insuficientes (tiene {len(df) if df is not None else 0}, necesita {lookback}).")
            return {}

        relevant_df = df.tail(lookback)
        
        # Usar el máximo high y mínimo low del periodo de lookback como el swing
        # Esto es una simplificación. Una detección de zigzag sería más precisa.
        swing_high_price = relevant_df['high'].max()
        swing_low_price = relevant_df['low'].min()
        
        if pd.isna(swing_high_price) or pd.isna(swing_low_price) or swing_high_price <= swing_low_price:
            logger.debug(f"Fibo: Swing inválido (H={swing_high_price}, L={swing_low_price}) o valores NaN.")
            return {}

        diff = swing_high_price - swing_low_price
        if diff == 0: # Evitar división por cero si high == low (aunque ya cubierto arriba)
            logger.debug("Fibo: Diferencia entre swing high y low es cero.")
            return {0.0: swing_low_price, 100.0: swing_high_price} # Solo los extremos

        # Niveles Fibo estándar: 0% es el inicio del swing (bajo), 100% es el final (alto).
        # Los retrocesos se miden DENTRO de este rango.
        fib_prices_for_ui = {
            0.0: swing_low_price,    # Base del swing
            100.0: swing_high_price, # Techo del swing
        }
        standard_levels_pct = [23.6, 38.2, 50.0, 61.8, 78.6] # Porcentajes
        
        for level_pct in standard_levels_pct:
            level_decimal = level_pct / 100.0
            # Para un swing de low a high, los niveles son: low + (diff * level_decimal)
            fib_prices_for_ui[level_pct] = swing_low_price + (diff * level_decimal)
            
        logger.debug(f"Fibonacci levels calculados (0%=low, 100%=high): {fib_prices_for_ui}")
        return fib_prices_for_ui