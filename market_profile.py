# market_profile.py
import pandas as pd
import numpy as np
from utils import get_logger
import config

logger = get_logger(__name__)

class MarketProfileAnalyzer:
    def __init__(self, calculator_instance):
        self.calculator = calculator_instance # Reutilizar el calculador de indicadores
        logger.info("MarketProfileAnalyzer inicializado.")

    def get_global_market_profile(self, df_global_market: pd.DataFrame):
        """
        Analiza el DataFrame del mercado global (ej. BTC/USDT) para determinar un perfil.
        Devuelve un diccionario con el perfil.
        """
        profile = {
            'overall_sentiment': 'NEUTRAL', # BULLISH, BEARISH, NEUTRAL
            'volatility_level': 'NORMAL', # LOW, NORMAL, HIGH, EXTREME
            'trend_strength': 'WEAK',     # WEAK, MODERATE, STRONG
            'regime': 'RANGING'           # TRENDING_UP, TRENDING_DOWN, RANGING, CHOPPY
        }
        if df_global_market is None or df_global_market.empty or len(df_global_market) < config.EMA_L_P:
            logger.warning("Datos insuficientes para el perfil del mercado global.")
            return profile

        # Usar el calculador para añadir indicadores necesarios al df_global_market
        # Asegúrate que el df_global_market que se pasa aquí ya tenga los indicadores necesarios
        # o llámalos explícitamente. Para este ejemplo, asumimos que ya tiene algunos.
        # Si no, deberías hacer: df_global_market = self.calculator.add_indicators(df_global_market.copy())
        # pero solo los necesarios para el perfil.

        # Volatilidad (ATR Normalizado)
        atr_col = f'ATR_{config.ATR_P}'
        if atr_col in df_global_market.columns and 'close' in df_global_market.columns:
            atr_val = df_global_market[atr_col].iloc[-1]
            close_val = df_global_market['close'].iloc[-1]
            if not pd.isna(atr_val) and not pd.isna(close_val) and close_val > 0:
                norm_atr = (atr_val / close_val) * 100
                if norm_atr > 3.5: profile['volatility_level'] = 'EXTREME'
                elif norm_atr > 1.8: profile['volatility_level'] = 'HIGH'
                elif norm_atr < 0.6: profile['volatility_level'] = 'LOW'
        else:
            logger.warning("ATR o Cierre no disponible para perfil de volatilidad global.")


        # Fuerza de Tendencia (ADX)
        adx_col = f'ADX_{config.ADX_P}'
        if adx_col in df_global_market.columns:
            adx_val = df_global_market[adx_col].iloc[-1]
            if not pd.isna(adx_val):
                if adx_val > 35: profile['trend_strength'] = 'STRONG' # ADX más alto para tendencia global
                elif adx_val > 20: profile['trend_strength'] = 'MODERATE'
        else:
            logger.warning("ADX no disponible para perfil de fuerza de tendencia global.")

        # Sentimiento General y Régimen (basado en EMAs sobre el precio global)
        ema_s_col = f'EMA_{config.MARKET_REGIME_EMA_SHORT}'
        ema_l_col = f'EMA_{config.MARKET_REGIME_EMA_LONG}'
        
        # Asegurarse de que estas EMAs específicas se calculen si no están
        if ema_s_col not in df_global_market.columns:
            df_global_market.ta.ema(length=config.MARKET_REGIME_EMA_SHORT, append=True, col_names=(ema_s_col,))
        if ema_l_col not in df_global_market.columns:
            df_global_market.ta.ema(length=config.MARKET_REGIME_EMA_LONG, append=True, col_names=(ema_l_col,))

        close_val = df_global_market['close'].iloc[-1] if 'close' in df_global_market.columns else np.nan
        ema_s_val = df_global_market[ema_s_col].iloc[-1] if ema_s_col in df_global_market.columns else np.nan
        ema_l_val = df_global_market[ema_l_col].iloc[-1] if ema_l_col in df_global_market.columns else np.nan

        if not any(pd.isna(v) for v in [close_val, ema_s_val, ema_l_val]):
            if close_val > ema_s_val and ema_s_val > ema_l_val:
                profile['overall_sentiment'] = 'BULLISH'
                if profile['trend_strength'] in ['MODERATE', 'STRONG']:
                    profile['regime'] = 'TRENDING_UP'
                else:
                    profile['regime'] = 'CHOPPY_UP' # Tendencia débil pero alcista
            elif close_val < ema_s_val and ema_s_val < ema_l_val:
                profile['overall_sentiment'] = 'BEARISH'
                if profile['trend_strength'] in ['MODERATE', 'STRONG']:
                    profile['regime'] = 'TRENDING_DOWN'
                else:
                    profile['regime'] = 'CHOPPY_DOWN'
            else: # EMAs cruzadas o precio entre ellas
                profile['overall_sentiment'] = 'NEUTRAL'
                if profile['trend_strength'] == 'WEAK':
                     profile['regime'] = 'RANGING'
                else: # ADX > 20 pero EMAs no alineadas -> CHOPPY
                     profile['regime'] = 'CHOPPY'
        else:
             logger.warning("EMAs o Cierre no disponibles para perfil de sentimiento/régimen global.")
        
        logger.info(f"Perfil del Mercado Global: Sentimiento={profile['overall_sentiment']}, Régimen={profile['regime']}, Volatilidad={profile['volatility_level']}, FuerzaTend={profile['trend_strength']}")
        return profile