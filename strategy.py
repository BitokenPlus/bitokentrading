# strategy.py
import pandas as pd
import numpy as np
from utils import get_logger
import config
import math # Para isnan

logger = get_logger(__name__)

class TradingStrategy:
    def __init__(self):
        logger.info("TradingStrategy (Avanzada v2) inicializada.")
        # Cargar umbrales y parámetros desde config para fácil ajuste
        self.adx_threshold = config.ADX_TREND_THRESHOLD
        self.volume_spike_factor = config.VOLUME_SPIKE_FACTOR
        self.oi_high_threshold = config.OI_HIGH_THRESHOLD_USD
        self.signal_threshold = config.SIGNAL_CONFIDENCE_THRESHOLD
        
        # Parámetros de indicadores para nombres de columna consistentes
        self.rsi_p = config.RSI_PERIOD
        self.macd_f, self.macd_s, self.macd_sig = config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
        self.sma_short_p, self.sma_long_p = config.SMA_SHORT_PERIOD, config.SMA_LONG_PERIOD
        self.ema_short_p, self.ema_long_p = config.EMA_SHORT_PERIOD, config.EMA_LONG_PERIOD
        self.bb_p, self.bb_std = config.BB_PERIOD, config.BB_STD_DEV
        self.adx_p_val = config.ADX_PERIOD # Renombrado para evitar conflicto con self.adx_threshold
        self.atr_p_val = config.ATR_PERIOD
        self.vol_ma_p = config.VOLUME_MA_PERIOD

    def _s(self, value, precision=2): # Helper para formatear floats o devolver N/A
        if pd.isna(value) or (isinstance(value, float) and math.isnan(value)):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{precision}f}"
        return str(value)

    def _get_indicator_val(self, df, col_name_base, *args, default=np.nan):
        """Helper para obtener el último valor de una columna de indicador."""
        # Construir el nombre de la columna basado en la convención de pandas_ta
        # Ej: 'RSI_14', 'MACD_12_26_9', 'EMA_20'
        parts = [col_name_base] + [str(a) for a in args]
        col_name = '_'.join(parts)
        
        if col_name in df.columns and not df[col_name].empty:
            val = df[col_name].iloc[-1]
            return val if not pd.isna(val) else default # Devolver np.nan si es NaN para cálculos
        # logger.warning(f"Columna '{col_name}' no encontrada o vacía en DataFrame para obtener indicador.")
        return default

    def analyze_pair(self, df_ltf, df_htf, pair_symbol, open_interest_data=None, funding_rate_data=None):
        analysis = {
            'pair': pair_symbol, 'signal': 'NEUTRAL', 'confidence': 0.0, 'reasoning': [],
            'trend_ltf': 'Indeterminado', 'trend_htf': 'Indeterminado', 
            'volatility_status': 'Normal', 'market_phase_ltf': 'Indeterminado', # Podría ser más avanzado
            'indicators': {}, 'fib_levels': {}, 'support': np.nan, 'resistance': np.nan,
            'open_interest_value': np.nan, 'funding_rate': np.nan, 'next_funding_time': None,
            'df_ltf': df_ltf, # Guardar para el gráfico
            'error': None
        }

        if df_ltf is None or df_ltf.empty or df_htf is None or df_htf.empty:
            analysis['error'] = "Datos OHLCV insuficientes para el análisis."
            logger.warning(f"{pair_symbol}: {analysis['error']}")
            return analysis

        # --- 1. Extracción de Indicadores ---
        ind = analysis['indicators'] # Shorthand
        try:
            # LTF
            ind['Close_LTF'] = self._get_indicator_val(df_ltf, 'close')
            ind['RSI_LTF'] = self._get_indicator_val(df_ltf, 'RSI', self.rsi_p)
            ind['MACD_LTF'] = self._get_indicator_val(df_ltf, 'MACD', self.macd_f, self.macd_s, self.macd_sig)
            ind['MACD_Signal_LTF'] = self._get_indicator_val(df_ltf, 'MACDs', self.macd_f, self.macd_s, self.macd_sig)
            ind['SMA_short_LTF'] = self._get_indicator_val(df_ltf, 'SMA', self.sma_short_p)
            ind['SMA_long_LTF'] = self._get_indicator_val(df_ltf, 'SMA', self.sma_long_p)
            ind['EMA_short_LTF'] = self._get_indicator_val(df_ltf, 'EMA', self.ema_short_p)
            ind['EMA_long_LTF'] = self._get_indicator_val(df_ltf, 'EMA', self.ema_long_p)
            ind['ADX_LTF'] = self._get_indicator_val(df_ltf, 'ADX', self.adx_p_val)
            ind['ATR_LTF'] = self._get_indicator_val(df_ltf, 'ATR', self.atr_p_val)
            ind['BBU_LTF'] = self._get_indicator_val(df_ltf, 'BBU', self.bb_p, self.bb_std)
            ind['BBL_LTF'] = self._get_indicator_val(df_ltf, 'BBL', self.bb_p, self.bb_std)
            ind['Volume_LTF'] = self._get_indicator_val(df_ltf, 'volume')
            ind['Volume_MA_LTF'] = self._get_indicator_val(df_ltf, 'volume_ma', self.vol_ma_p)
            
            # HTF
            ind['Close_HTF'] = self._get_indicator_val(df_htf, 'close') # Necesario para comparar con EMA HTF
            ind['EMA_long_HTF'] = self._get_indicator_val(df_htf, 'EMA', self.ema_long_p)
            ind['RSI_HTF'] = self._get_indicator_val(df_htf, 'RSI', self.rsi_p)
            ind['ADX_HTF'] = self._get_indicator_val(df_htf, 'ADX', self.adx_p_val)

            for col_name in df_ltf.columns:
                if col_name.startswith('fib_'):
                    try:
                        level_pct = float(col_name.split('_')[1])
                        val = self._get_indicator_val(df_ltf, col_name) # Usa el helper
                        if not pd.isna(val): analysis['fib_levels'][level_pct] = val
                    except (ValueError, IndexError):
                        logger.debug(f"No se pudo parsear nivel Fibo desde columna '{col_name}'.")
            analysis['support'] = analysis['fib_levels'].get(0.0, np.nan)
            analysis['resistance'] = analysis['fib_levels'].get(100.0, np.nan)

        except Exception as e:
            analysis['error'] = f"Error crítico extrayendo indicadores: {e}"
            logger.error(f"{pair_symbol}: {analysis['error']}", exc_info=True)
            return analysis
        
        # --- 2. Interpretación de Métricas de Futuros ---
        if open_interest_data:
            oi_val = open_interest_data.get('openInterestValue')
            if oi_val is None:
                 oi_amount = open_interest_data.get('openInterestAmount')
                 if oi_amount is not None and not pd.isna(ind['Close_LTF']):
                     oi_val = oi_amount * ind['Close_LTF']
            analysis['open_interest_value'] = oi_val if oi_val is not None else np.nan

        if funding_rate_data:
            analysis['funding_rate'] = funding_rate_data.get('fundingRate', np.nan)
            analysis['next_funding_time'] = funding_rate_data.get('fundingTimestamp')

        # --- 3. Análisis de Tendencia y Volatilidad (con chequeo de NaNs) ---
        # HTF Trend
        close_ltf, ema_l_htf, adx_htf_val = ind['Close_LTF'], ind['EMA_long_HTF'], ind['ADX_HTF']
        if not any(pd.isna(v) for v in [close_ltf, ema_l_htf, adx_htf_val]):
            if close_ltf > ema_l_htf and adx_htf_val > self.adx_threshold: analysis['trend_htf'] = "Alcista Fuerte"
            elif close_ltf < ema_l_htf and adx_htf_val > self.adx_threshold: analysis['trend_htf'] = "Bajista Fuerte"
            elif close_ltf > ema_l_htf: analysis['trend_htf'] = "Alcista Débil"
            else: analysis['trend_htf'] = "Bajista Débil" # Asumir bajista si no es alcista y no hay datos para lateral
            analysis['reasoning'].append(f"HTF: {analysis['trend_htf']} (P:{self._s(close_ltf)} vs EMA{self.ema_long_p}:{self._s(ema_l_htf)}, ADX:{self._s(adx_htf_val,1)})")
        else: analysis['reasoning'].append("HTF: Indeterminado (datos insuficientes).")

        # LTF Trend
        macd_val, macds_val, sma_s, sma_l, adx_ltf_val = ind['MACD_LTF'], ind['MACD_Signal_LTF'], ind['SMA_short_LTF'], ind['SMA_long_LTF'], ind['ADX_LTF']
        if not any(pd.isna(v) for v in [macd_val, macds_val, sma_s, sma_l, adx_ltf_val]):
            if macd_val > macds_val and sma_s > sma_l and adx_ltf_val > self.adx_threshold: analysis['trend_ltf'] = "Alcista Fuerte"
            elif macd_val < macds_val and sma_s < sma_l and adx_ltf_val > self.adx_threshold: analysis['trend_ltf'] = "Bajista Fuerte"
            elif macd_val > macds_val or sma_s > sma_l : analysis['trend_ltf'] = "Alcista Débil"
            else: analysis['trend_ltf'] = "Bajista Débil"
            analysis['reasoning'].append(f"LTF: {analysis['trend_ltf']} (MACD, SMAs, ADX:{self._s(adx_ltf_val,1)})")
        else: analysis['reasoning'].append("LTF: Indeterminado (datos insuficientes).")
        
        # Volatility
        atr_val = ind['ATR_LTF']
        if not any(pd.isna(v) for v in [atr_val, close_ltf]) and close_ltf > 0:
            norm_atr = (atr_val / close_ltf) * 100
            if norm_atr > 4.0: analysis['volatility_status'] = "Extrema" # Ajustado umbral
            elif norm_atr > 2.0: analysis['volatility_status'] = "Alta"
            elif norm_atr < 0.7: analysis['volatility_status'] = "Baja"
            analysis['reasoning'].append(f"Volatilidad (ATR%): {self._s(norm_atr)}%, Estado: {analysis['volatility_status']}")

        # --- 4. Lógica de Señal (Scores) ---
        long_score, short_score = 0.0, 0.0
        rsi_val, bbl_val, bbu_val = ind['RSI_LTF'], ind['BBL_LTF'], ind['BBU_LTF']
        vol_val, vol_ma_val = ind['Volume_LTF'], ind['Volume_MA_LTF']
        fr_val = analysis['funding_rate']
        oi_val = analysis['open_interest_value']

        # Ponderaciones para cada factor (puedes ponerlas en config.py)
        w = {
            'htf_trend': 1.5, 'ltf_trend': 1.0, 'rsi': 0.8, 'volume': 0.5, 
            'bb_squeeze_break': 0.0, 'bb_reversion': 0.7, 'oi': 0.3, 'fr': 0.2,
            'fib_support': 0.6, 'fib_resistance': 0.6
        }
        
        # Long Conditions
        if "Alcista" in analysis['trend_htf']: long_score += w['htf_trend']
        if analysis['trend_ltf'] == "Alcista Fuerte": long_score += w['ltf_trend']
        if not pd.isna(rsi_val) and config.RSI_OVERSOLD < rsi_val < (config.RSI_OVERBOUGHT - 10): long_score += w['rsi']
        if not any(pd.isna(v) for v in [vol_val, vol_ma_val]) and vol_val > vol_ma_val * self.volume_spike_factor: long_score += w['volume']
        if "Alcista" in analysis['trend_htf'] and not any(pd.isna(v) for v in [close_ltf, bbl_val]) and close_ltf <= bbl_val * 1.01: long_score += w['bb_reversion'] # Reversión desde BBL
        if not pd.isna(oi_val) and oi_val > self.oi_high_threshold : long_score += w['oi']
        if not pd.isna(fr_val):
            if fr_val <= 0.00035: long_score += w['fr'] # FR neutral o favorable
            elif fr_val > 0.0008: long_score -= w['fr'] * 1.5 # FR muy alto penaliza más
        # Considerar rebote en soporte Fibo (ej. 38.2% o 61.8%)
        if not pd.isna(analysis['support']) and not pd.isna(close_ltf) and analysis['fib_levels'].get(38.2) and \
           analysis['fib_levels'].get(38.2) * 0.99 < close_ltf < analysis['fib_levels'].get(38.2) * 1.015:
            long_score += w['fib_support']
            analysis['reasoning'].append(f"Rebote potencial en Fibo 38.2% ({self._s(analysis['fib_levels'][38.2])}).")


        # Short Conditions
        if "Bajista" in analysis['trend_htf']: short_score += w['htf_trend']
        if analysis['trend_ltf'] == "Bajista Fuerte": short_score += w['ltf_trend']
        if not pd.isna(rsi_val) and (config.RSI_OVERSOLD + 10) < rsi_val < config.RSI_OVERBOUGHT: short_score += w['rsi']
        if not any(pd.isna(v) for v in [vol_val, vol_ma_val]) and vol_val > vol_ma_val * self.volume_spike_factor: short_score += w['volume']
        if "Bajista" in analysis['trend_htf'] and not any(pd.isna(v) for v in [close_ltf, bbu_val]) and close_ltf >= bbu_val * 0.99: short_score += w['bb_reversion'] # Reversión desde BBU
        if not pd.isna(oi_val) and oi_val > self.oi_high_threshold : short_score += w['oi']
        if not pd.isna(fr_val):
            if fr_val >= -0.00035: short_score += w['fr']
            elif fr_val < -0.0008: short_score -= w['fr'] * 1.5
        if not pd.isna(analysis['resistance']) and not pd.isna(close_ltf) and analysis['fib_levels'].get(61.8) and \
           analysis['fib_levels'].get(61.8) * 0.985 < close_ltf < analysis['fib_levels'].get(61.8) * 1.01:
            short_score += w['fib_resistance']
            analysis['reasoning'].append(f"Rechazo potencial en Fibo 61.8% ({self._s(analysis['fib_levels'][61.8])}).")


        max_possible_score = sum(w.values()) # Suma de todas las ponderaciones positivas posibles (aproximado)
        
        conf_l = (long_score / max_possible_score) if max_possible_score > 0 else 0
        conf_s = (short_score / max_possible_score) if max_possible_score > 0 else 0
        
        analysis['confidence'] = 0.0 # Default
        if conf_l > self.signal_threshold and conf_l > (conf_s + 0.05): # Long tiene ventaja clara
            analysis['signal'] = 'LONG'
            analysis['confidence'] = round(conf_l, 2)
        elif conf_s > self.signal_threshold and conf_s > (conf_l + 0.05): # Short tiene ventaja clara
            analysis['signal'] = 'SHORT'
            analysis['confidence'] = round(conf_s, 2)
        else: # Neutral o señales conflictivas/débiles
            analysis['signal'] = 'NEUTRAL'
            # Mostrar la confianza del lado más fuerte, incluso si es neutral
            if conf_l > conf_s: analysis['confidence'] = round(conf_l, 2)
            else: analysis['confidence'] = round(conf_s, 2)
        
        analysis['reasoning'].insert(0, f"DECISIÓN: {analysis['signal']} (Conf: {analysis['confidence']:.0%}, LRaw:{long_score:.1f}, SRaw:{short_score:.1f})")

        if analysis['volatility_status'] == "Extrema" and analysis['signal'] not in ['NEUTRAL', 'ERROR']:
            analysis['signal'] = f"CAUTIOUS_{analysis['signal']}"
            analysis['reasoning'].append("MODIFICADOR: Volatilidad Extrema -> Señal Cautelosa.")
        elif analysis['volatility_status'] == "Baja" and analysis['signal'] not in ['NEUTRAL', 'ERROR']:
            analysis['reasoning'].append("INFO: Volatilidad Baja, considerar trades de rango o esperar ruptura.")
        
        analysis['reasoning'] = analysis['reasoning'][:12]

        logger.info(
            f"Análisis {pair_symbol}: Sig={analysis['signal']} (Conf:{analysis['confidence']:.0%}), "
            f"Tend_HTF={analysis['trend_htf']}, Tend_LTF={analysis['trend_ltf']}, "
            f"RSI={self._s(ind['RSI_LTF'],1)}, ADX={self._s(ind['ADX_LTF'],1)}, "
            f"OI={self._s(analysis['open_interest_value'],0)}, FR={self._s(analysis['funding_rate']*100 if not pd.isna(analysis['funding_rate']) else np.nan,4)}%"
        )
        return analysis