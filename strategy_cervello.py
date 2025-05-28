# strategy_cervello.py
import pandas as pd
import numpy as np
from utils import get_logger
import config # Importa tu archivo de configuración completo
import math # Para isnan y otras operaciones

logger = get_logger(__name__)

class CervelloStrategy:
    def __init__(self):
        logger.info(f"CervelloStrategy ({getattr(config, 'APP_VERSION', 'N/A')}) inicializada.")
        self.params = config 
        self.weights = getattr(config, 'CERVELLO_WEIGHTS', {}) 

        # Parámetros de indicadores
        self.rsi_p = getattr(config, 'RSI_PERIOD', 14)
        self.rsi_ob = getattr(config, 'RSI_OB', 70)
        self.rsi_os = getattr(config, 'RSI_OS', 30)
        self.macd_f = getattr(config, 'MACD_F', 12); self.macd_s = getattr(config, 'MACD_S', 26); self.macd_sig = getattr(config, 'MACD_SIG', 9)
        self.sma_s_p = getattr(config, 'SMA_S_P', 20); self.sma_l_p = getattr(config, 'SMA_L_P', 50)
        self.ema_s_p = getattr(config, 'EMA_S_P', 12); self.ema_l_p = getattr(config, 'EMA_L_P', 26)
        self.bb_p = getattr(config, 'BB_P', 20); self.bb_std = getattr(config, 'BB_STD', 2.0)
        self.adx_p_val = getattr(config, 'ADX_P', 14)
        self.atr_p_val = getattr(config, 'ATR_P', 14)
        self.vol_ma_p = getattr(config, 'VOL_MA_P', 20)
        self.ichimoku_t = getattr(config, 'ICHIMOKU_TENKAN', 9)
        self.ichimoku_k = getattr(config, 'ICHIMOKU_KIJUN', 26)
        self.ichimoku_sb = getattr(config, 'ICHIMOKU_SENKOU_B', 52)
        
        self.sl_atr_multiplier = getattr(config, 'SL_ATR_MULTIPLIER', 1.5)
        self.tp1_atr_multiplier = getattr(config, 'TP1_ATR_MULTIPLIER', 1.0)
        self.tp2_atr_multiplier = getattr(config, 'TP2_ATR_MULTIPLIER', 2.0)
        self.tp3_atr_multiplier = getattr(config, 'TP3_ATR_MULTIPLIER', 3.5)
        self.min_rrr = getattr(config, 'MIN_RISK_REWARD_RATIO', 1.5) # Mínimo RRR para TP1

    def _s(self, value, precision=2): # Formateador de string
        if pd.isna(value) or (isinstance(value, float) and math.isnan(value)): return "N/A"
        if isinstance(value, (float, np.floating)): return f"{value:.{precision}f}"
        return str(value)

    def _get_ind(self, df: pd.DataFrame, name_parts: tuple, default=np.nan): # Obtener indicador
        if not isinstance(name_parts, tuple): name_parts = (name_parts,)
        col_name = '_'.join(map(str, name_parts))
        if col_name in df.columns and not df[col_name].empty:
            val = df[col_name].iloc[-1]
            return val if not pd.isna(val) else default
        logger.debug(f"Columna '{col_name}' no encontrada o vacía en _get_ind.")
        return default
    
    def _detect_rsi_divergence(self, df_ltf: pd.DataFrame, rsi_col_name_str: str, lookback:int=20):
        # ... (Placeholder - Mantener la lógica de detección de divergencia como no implementada robustamente por ahora)
        return None 

    def _get_ichimoku_signal(self, df_ltf: pd.DataFrame):
        # ... (Lógica de Ichimoku como en la v2.4, asegurando que los nombres de columna sean correctos)
        isa_col = f'ISA_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}'
        isb_col = f'ISB_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}'
        its_col = f'ITS_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}'
        iks_col = f'IKS_{self.ichimoku_t}_{self.ichimoku_k}_{self.ichimoku_sb}'
        
        required_cols = [isa_col, isb_col, its_col, iks_col, 'close']
        if not all(col in df_ltf.columns for col in required_cols):
            logger.debug(f"Faltan columnas de Ichimoku para señal. Requeridas: {required_cols}, Presentes: {list(df_ltf.columns)}")
            return "ICHIMOKU_DATA_MISSING"

        close = self._get_ind(df_ltf, ('close',))
        tenkan, kijun = self._get_ind(df_ltf, (its_col,)), self._get_ind(df_ltf, (iks_col,))
        senkou_a, senkou_b = self._get_ind(df_ltf, (isa_col,)), self._get_ind(df_ltf, (isb_col,))

        if any(pd.isna(v) for v in [close, tenkan, kijun, senkou_a, senkou_b]):
            return "ICHIMOKU_NAN_VALUES"

        price_above_kumo = close > max(senkou_a, senkou_b)
        price_below_kumo = close < min(senkou_a, senkou_b)
        kumo_bullish, kumo_bearish = senkou_a > senkou_b, senkou_a < senkou_b
        tk_cross_bullish, tk_cross_bearish = tenkan > kijun, tenkan < kijun
        
        if price_above_kumo and kumo_bullish and tk_cross_bullish and tenkan > max(senkou_a, senkou_b):
            return "STRONG_BULLISH_ICHIMOKU"
        if price_below_kumo and kumo_bearish and tk_cross_bearish and tenkan < min(senkou_a, senkou_b):
            return "STRONG_BEARISH_ICHIMOKU"
        return "NEUTRAL_ICHIMOKU"


    def _calculate_price_targets_stops(self, signal, entry_price, fib_levels, atr_val):
        targets_stops = {'tp1': np.nan, 'tp2': np.nan, 'tp3': np.nan, 'sl': np.nan}
        if pd.isna(entry_price) or pd.isna(atr_val) or atr_val <= 1e-9:
            logger.debug(f"No se pueden calcular TP/SL: entrada={entry_price}, atr={atr_val}")
            return targets_stops

        # Stop Loss
        if signal.startswith("LONG"): targets_stops['sl'] = entry_price - (self.sl_atr_multiplier * atr_val)
        elif signal.startswith("SHORT"): targets_stops['sl'] = entry_price + (self.sl_atr_multiplier * atr_val)
        
        # Take Profits ATR
        if signal.startswith("LONG"):
            targets_stops['tp1'] = entry_price + (self.tp1_atr_multiplier * atr_val)
            targets_stops['tp2'] = entry_price + (self.tp2_atr_multiplier * atr_val)
            targets_stops['tp3'] = entry_price + (self.tp3_atr_multiplier * atr_val)
        elif signal.startswith("SHORT"):
            targets_stops['tp1'] = entry_price - (self.tp1_atr_multiplier * atr_val)
            targets_stops['tp2'] = entry_price - (self.tp2_atr_multiplier * atr_val)
            targets_stops['tp3'] = entry_price - (self.tp3_atr_multiplier * atr_val)

        # Refinar TP3 con Fibo
        if fib_levels and isinstance(fib_levels, dict):
            fib_100 = fib_levels.get(100.0, np.nan)
            fib_0 = fib_levels.get(0.0, np.nan)
            if signal.startswith("LONG") and not pd.isna(fib_100) and fib_100 > entry_price:
                if pd.isna(targets_stops['tp3']) or abs(fib_100 - entry_price) > abs(targets_stops.get('tp3', entry_price) - entry_price):
                     targets_stops['tp3'] = fib_100
            elif signal.startswith("SHORT") and not pd.isna(fib_0) and fib_0 < entry_price:
                if pd.isna(targets_stops['tp3']) or abs(fib_0 - entry_price) > abs(targets_stops.get('tp3', entry_price) - entry_price):
                    targets_stops['tp3'] = fib_0
        
        # Validar RRR para TP1
        sl, tp1 = targets_stops.get('sl', np.nan), targets_stops.get('tp1', np.nan)
        if not any(pd.isna(v) for v in [sl, tp1, entry_price]):
            profit = abs(tp1 - entry_price)
            loss = abs(entry_price - sl)
            if loss > 1e-9:
                rrr = profit / loss
                if rrr < self.min_rrr: logger.debug(f"RRR TP1 ({rrr:.2f}) < min ({self.min_rrr}).")
            else: logger.debug("SL muy cerca, RRR no calculable.")

        # Limpieza final
        for k, v in list(targets_stops.items()): # Usar list() para iterar sobre copia
            if pd.isna(v) or v <= 0: targets_stops[k] = np.nan; continue
            if k.startswith('tp'):
                if (signal.startswith("LONG") and v < entry_price * 1.001) or \
                   (signal.startswith("SHORT") and v > entry_price * 0.999):
                    targets_stops[k] = np.nan
            elif k == 'sl':
                if (signal.startswith("LONG") and v > entry_price) or \
                   (signal.startswith("SHORT") and v < entry_price):
                    targets_stops[k] = np.nan
        return targets_stops

    def analyze_pair(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame, pair_symbol: str, 
                     market_profile: dict, 
                     open_interest_data=None, funding_rate_data=None):
        
        analysis = {
            'pair': pair_symbol, 'signal': 'NEUTRAL', 'confidence': 0.0, 'reasoning': [],
            'trend_ltf': 'Indeterminado', 'trend_htf': 'Indeterminado', 
            'volatility_status_pair': 'Normal',
            'indicators': {}, 'fib_levels': {}, 'support': np.nan, 'resistance': np.nan,
            'open_interest_value': np.nan, 'funding_rate': np.nan, 'next_funding_time': None,
            'df_ltf': df_ltf, 'error': None,
            'market_profile_applied': market_profile,
            'price_targets': {'tp1': np.nan, 'tp2': np.nan, 'tp3': np.nan, 'sl': np.nan}
        }

        min_len_ltf = max(self.params.FIB_LOOKBACK, self.sma_l_p, self.ema_l_p, self.bb_p, self.adx_p_val, self.atr_p_val, self.ichimoku_sb) + 10 # +buffer
        min_len_htf = max(self.ema_l_p, self.adx_p_val, self.rsi_p) + 10
        
        if df_ltf is None or df_ltf.empty or len(df_ltf) < min_len_ltf :
            analysis['error'] = f"Datos OHLCV LTF insuficientes ({len(df_ltf) if df_ltf is not None else 0} < {min_len_ltf} velas)."
            return analysis # No continuar si no hay suficientes datos base
        if df_htf is None or df_htf.empty or len(df_htf) < min_len_htf :
            analysis['error'] = f"Datos OHLCV HTF insuficientes ({len(df_htf) if df_htf is not None else 0} < {min_len_htf} velas)."
            return analysis

        ind = analysis['indicators']
        try: # Extracción de Indicadores
            ind['Close_LTF'] = self._get_ind(df_ltf, ('close',))
            rsi_col_name_str = f'RSI_{self.rsi_p}'
            ind['RSI_LTF'] = self._get_ind(df_ltf, (rsi_col_name_str,))
            ind['MACD_LTF'] = self._get_ind(df_ltf, ('MACD', self.macd_f, self.macd_s, self.macd_sig))
            ind['MACD_Signal_LTF'] = self._get_ind(df_ltf, ('MACDs', self.macd_f, self.macd_s, self.macd_sig))
            ind['MACD_Hist_LTF'] = self._get_ind(df_ltf, ('MACDh', self.macd_f, self.macd_s, self.macd_sig)) # Añadido
            ind['SMA_short_LTF'] = self._get_ind(df_ltf, ('SMA', self.sma_s_p))
            ind['SMA_long_LTF'] = self._get_ind(df_ltf, ('SMA', self.sma_l_p))
            ind['EMA_short_LTF'] = self._get_ind(df_ltf, ('EMA', self.ema_s_p)) # Añadido
            ind['EMA_long_LTF'] = self._get_ind(df_ltf, ('EMA', self.ema_l_p))
            ind['ADX_LTF'] = self._get_ind(df_ltf, ('ADX', self.adx_p_val))
            ind['ATR_LTF'] = self._get_ind(df_ltf, ('ATR', self.atr_p_val))
            ind['BBU_LTF'] = self._get_ind(df_ltf, ('BBU', self.bb_p, self.bb_std))
            ind['BBL_LTF'] = self._get_ind(df_ltf, ('BBL', self.bb_p, self.bb_std))
            ind['Volume_LTF'] = self._get_ind(df_ltf, ('volume',))
            ind['Volume_MA_LTF'] = self._get_ind(df_ltf, ('volume_ma', self.vol_ma_p))
            
            ind['Close_HTF'] = self._get_ind(df_htf, ('close',))
            ind['EMA_long_HTF'] = self._get_ind(df_htf, ('EMA', self.ema_l_p))
            ind['RSI_HTF'] = self._get_ind(df_htf, (f'RSI_{self.rsi_p}',))
            ind['ADX_HTF'] = self._get_ind(df_htf, ('ADX', self.adx_p_val))

            for col_name_fib in df_ltf.columns: # Renombrado para evitar colisión
                if col_name_fib.startswith('fib_'):
                    try:
                        level_pct_fib = float(col_name_fib.split('_')[1])
                        val_fib = self._get_ind(df_ltf, (col_name_fib,))
                        if not pd.isna(val_fib): analysis['fib_levels'][level_pct_fib] = val_fib
                    except (ValueError, IndexError): logger.debug(f"Error parseando Fibo '{col_name_fib}'")
            analysis['support'] = analysis['fib_levels'].get(0.0, np.nan)
            analysis['resistance'] = analysis['fib_levels'].get(100.0, np.nan)
        except Exception as e_ind:
            analysis['error'] = f"Error extrayendo indicadores: {e_ind}"
            logger.error(f"Error en {pair_symbol} extrayendo ind: {e_ind}", exc_info=True)
            return analysis
        
        # Interpretación de Métricas de Futuros
        if open_interest_data:
            # ... (lógica OI como antes) ...
            oi_v = open_interest_data.get('openInterestValue')
            if oi_v is None or math.isnan(oi_v):
                 oi_a = open_interest_data.get('openInterestAmount')
                 if oi_a is not None and not math.isnan(oi_a) and not pd.isna(ind['Close_LTF']) and ind['Close_LTF'] > 0:
                     oi_v = oi_a * ind['Close_LTF'] 
            analysis['open_interest_value'] = oi_v if oi_v is not None and not math.isnan(oi_v) else np.nan
        if funding_rate_data:
            analysis['funding_rate'] = funding_rate_data.get('fundingRate', np.nan)
            analysis['next_funding_time'] = funding_rate_data.get('fundingTimestamp')

        # --- "IA" LÓGICA DE DECISIÓN (Factores y Puntuación) ---
        long_score, short_score = 0.0, 0.0
        
        # Factor: Perfil General del Mercado
        profile_sentiment = market_profile.get('overall_sentiment', 'NEUTRAL')
        profile_regime = market_profile.get('regime', 'RANGING')
        profile_vol_global = market_profile.get('volatility_level', 'NORMAL') # Renombrado para claridad
        analysis['reasoning'].append(f"GlobalMkt: Sent={profile_sentiment}, Rég={profile_regime}, Vol={profile_vol_global}")
        if profile_sentiment == 'BULLISH': long_score += self.weights.get('market_regime', 2.0)
        elif profile_sentiment == 'BEARISH': short_score += self.weights.get('market_regime', 2.0)

        # Factor: Tendencia HTF
        close_ltf, ema_l_htf, adx_htf = ind['Close_LTF'], ind['EMA_long_HTF'], ind['ADX_HTF']
        htf_trend_reason = "HTF: Indeterminado (datos NaN)"
        if not any(pd.isna(v) for v in [close_ltf, ema_l_htf, adx_htf]): # Si todos los datos están OK
            if close_ltf > ema_l_htf: 
                analysis['trend_htf'] = "Alcista Débil"; score_add = self.weights.get('htf_trend_weak', 0.8)
                if adx_htf > self.params.CERVELLO_ADX_THRESHOLD:
                    analysis['trend_htf'] = "Alcista Fuerte"; score_add = self.weights.get('htf_trend_strong', 1.8)
                long_score += score_add
            elif close_ltf < ema_l_htf:
                analysis['trend_htf'] = "Bajista Débil"; score_add = self.weights.get('htf_trend_weak', 0.8)
                if adx_htf > self.params.CERVELLO_ADX_THRESHOLD:
                    analysis['trend_htf'] = "Bajista Fuerte"; score_add = self.weights.get('htf_trend_strong', 1.8)
                short_score += score_add
            else: analysis['trend_htf'] = "Lateral"
            htf_trend_reason = f"HTF: {analysis['trend_htf']} (ADX {self._s(adx_htf,1)})"
        analysis['reasoning'].append(htf_trend_reason)
        
        # Factor: Tendencia LTF
        macd_ltf, macds_ltf, sma_s_ltf, sma_l_ltf, adx_ltf = ind['MACD_LTF'], ind['MACD_Signal_LTF'], ind['SMA_short_LTF'], ind['SMA_long_LTF'], ind['ADX_LTF']
        ltf_trend_reason = "LTF: Indeterminado (datos NaN)"
        if not any(pd.isna(v) for v in [macd_ltf, macds_ltf, sma_s_ltf, sma_l_ltf, adx_ltf]):
            if macd_ltf > macds_ltf and sma_s_ltf > sma_l_ltf: # Confirmación de ambas MAs y MACD
                analysis['trend_ltf'] = "Alcista Débil"; score_add = self.weights.get('ltf_trend_weak', 0.7)
                if adx_ltf > self.params.CERVELLO_ADX_THRESHOLD: # Si ADX confirma fuerza
                    analysis['trend_ltf'] = "Alcista Fuerte"; score_add = self.weights.get('ltf_trend_strong', 1.5)
                long_score += score_add
            elif macd_ltf < macds_ltf and sma_s_ltf < sma_l_ltf:
                analysis['trend_ltf'] = "Bajista Débil"; score_add = self.weights.get('ltf_trend_weak', 0.7)
                if adx_ltf > self.params.CERVELLO_ADX_THRESHOLD:
                    analysis['trend_ltf'] = "Bajista Fuerte"; score_add = self.weights.get('ltf_trend_strong', 1.5)
                short_score += score_add
            # Podrías añadir casos para MACD alcista pero SMA no, etc. (más granular)
            else: analysis['trend_ltf'] = "Lateral"
            ltf_trend_reason = f"LTF: {analysis['trend_ltf']} (ADX {self._s(adx_ltf,1)})"
        analysis['reasoning'].append(ltf_trend_reason)

        # Factor: Volatilidad del Par
        atr_val_pair = ind['ATR_LTF']
        if not any(pd.isna(v) for v in [atr_val_pair, close_ltf]) and close_ltf > 0:
            norm_atr_pair = (atr_val_pair / close_ltf) * 100
            if norm_atr_pair > 4.5: analysis['volatility_status_pair'] = "Extrema"
            elif norm_atr_pair > 2.2: analysis['volatility_status_pair'] = "Alta"
            elif norm_atr_pair < 0.8: analysis['volatility_status_pair'] = "Baja"
            analysis['reasoning'].append(f"Vol Par (ATR%): {self._s(norm_atr_pair)}% ({analysis['volatility_status_pair']})")

        # Factor: RSI LTF y Divergencias
        rsi_val, bbl_val, bbu_val = ind['RSI_LTF'], ind['BBL_LTF'], ind['BBU_LTF'] # BBL/BBU para lógica BB
        rsi_div_sig = self._detect_rsi_divergence(df_ltf, rsi_col_name_str, lookback=getattr(config, 'RSI_DIV_LOOKBACK', 20))
        if rsi_div_sig == "BULLISH_DIVERGENCE": long_score += self.weights.get('rsi_divergence', 1.5); analysis['reasoning'].append("RSI: Div. Alcista.")
        elif rsi_div_sig == "BEARISH_DIVERGENCE": short_score += self.weights.get('rsi_divergence', 1.5); analysis['reasoning'].append("RSI: Div. Bajista.")
        elif not pd.isna(rsi_val): # Si no hay divergencia, usar valor RSI
            if self.rsi_os < rsi_val < (self.rsi_ob - 10): long_score += self.weights.get('rsi_optimal', 1.0) # No sobrecomprado
            if (self.rsi_os + 10) < rsi_val < self.rsi_ob: short_score += self.weights.get('rsi_optimal', 1.0) # No sobrevendido
            analysis['reasoning'].append(f"RSI LTF: {self._s(rsi_val,1)}")
        
        # Factor: Ichimoku
        ichimoku_signal_val = self._get_ichimoku_signal(df_ltf)
        if ichimoku_signal_val == "STRONG_BULLISH_ICHIMOKU": 
            long_score += self.weights.get('ichimoku_strong_signal', 1.8)
        elif ichimoku_signal_val == "STRONG_BEARISH_ICHIMOKU": 
            short_score += self.weights.get('ichimoku_strong_signal', 1.8)
        if ichimoku_signal_val not in ["ICHIMOKU_DATA_MISSING", "ICHIMOKU_NAN_VALUES", "NEUTRAL_ICHIMOKU"]: # Loguear solo señales relevantes
            analysis['reasoning'].append(f"Ichimoku: {ichimoku_signal_val}")
        
        # ... (Añadir aquí la lógica COMPLETA para Volumen, OI, Bandas de Bollinger, Fibonacci usando self.weights)
        # EJEMPLO Volumen:
        vol_ltf_val, vol_ma_ltf_val = ind['Volume_LTF'], ind['Volume_MA_LTF']
        if not any(pd.isna(v) for v in [vol_ltf_val, vol_ma_ltf_val]) and vol_ma_ltf_val > 0: # Evitar división por cero
            if vol_ltf_val > vol_ma_ltf_val * self.params.CERVELLO_VOL_SPIKE_FACTOR:
                analysis['reasoning'].append(f"Volumen: Spike ({self._s(vol_ltf_val/vol_ma_ltf_val, 1)}x MA).")
                # El peso del volumen podría depender de si confirma la tendencia
                if "Alcista" in analysis['trend_ltf']: long_score += self.weights.get('volume_confirmation', 0.7)
                elif "Bajista" in analysis['trend_ltf']: short_score += self.weights.get('volume_confirmation', 0.7)
        
        # EJEMPLO Open Interest:
        oi_val_strat = analysis['open_interest_value'] # Usar el valor ya extraído
        if not pd.isna(oi_val_strat) and oi_val_strat > self.params.CERVELLO_OI_HIGH_THRESHOLD:
            analysis['reasoning'].append(f"OI: Alto ({self._s(oi_val_strat,0)} USD).")
            # Lógica más avanzada podría ver si OI aumenta con la tendencia
            long_score += self.weights.get('oi_confirmation', 0.5) # Ejemplo genérico, refinar
            short_score += self.weights.get('oi_confirmation', 0.5)


        # EJEMPLO Funding Rate:
        fr_val_strat = analysis['funding_rate']
        if not pd.isna(fr_val_strat):
            if -0.0002 <= fr_val_strat <= 0.0002: analysis['reasoning'].append(f"FR: Neutral ({self._s(fr_val_strat*100,4)}%).")
            elif fr_val_strat > 0.00075: # Muy positivo
                long_score += self.weights.get('funding_rate_contrarian', -0.5) 
                short_score += self.weights.get('funding_rate_favorable', 0.3) 
                analysis['reasoning'].append(f"FR: Alto Positivo ({self._s(fr_val_strat*100,4)}%).")
            elif fr_val_strat < -0.00075: # Muy negativo
                short_score += self.weights.get('funding_rate_contrarian', -0.5)
                long_score += self.weights.get('funding_rate_favorable', 0.3)
                analysis['reasoning'].append(f"FR: Alto Negativo ({self._s(fr_val_strat*100,4)}%).")
            else: # Funding moderado
                if fr_val_strat > 0 : long_score += self.weights.get('funding_rate_favorable', 0.3) / 2 # Menos peso
                if fr_val_strat < 0 : short_score += self.weights.get('funding_rate_favorable', 0.3) / 2


        # --- Decisión Final ---
        total_pos_w = sum(abs(w) for k, w in self.weights.items() if not k.endswith("contrarian") and isinstance(w, (int, float)) and w > 0)
        total_pos_w = max(total_pos_w, 1.0) 
        conf_l = min(1.0, max(0.0, (long_score / total_pos_w))); conf_s = min(1.0, max(0.0, (short_score / total_pos_w)))
        
        sig_thresh = self.params.CERVELLO_SIGNAL_CONFIDENCE_THRESHOLD
        global_vol_level = market_profile.get('volatility_level', 'NORMAL')
        if global_vol_level == 'EXTREME': sig_thresh = min(0.95, sig_thresh * 1.15)
        elif global_vol_level == 'LOW' and profile_regime == 'RANGING': sig_thresh = max(0.30, sig_thresh * 0.85)
        if global_vol_level != 'NORMAL' or (global_vol_level == 'LOW' and profile_regime == 'RANGING'): # Log si se ajusta
            analysis['reasoning'].append(f"Ajuste: Umbral señal -> {sig_thresh:.0%} (VolGlobal: {global_vol_level}, Rég: {profile_regime})")
        
        if conf_l >= sig_thresh and conf_l > (conf_s + 0.08): analysis['signal'] = 'LONG'
        elif conf_s >= sig_thresh and conf_s > (conf_l + 0.08): analysis['signal'] = 'SHORT'
        else: analysis['signal'] = 'NEUTRAL'
        analysis['confidence'] = round(max(conf_l, conf_s) if analysis['signal'] == 'NEUTRAL' else (conf_l if analysis['signal']=='LONG' else conf_s), 2)
        
        if analysis['volatility_status_pair'] == "Extrema" and analysis['signal'] not in ['NEUTRAL', 'ERROR']:
            analysis['signal'] = f"CAUTIOUS_{analysis['signal']}"
            analysis['reasoning'].append(f"MODIFICADOR (Par): Vol. Par Extrema -> Señal Cautelosa.")
        
        # Calcular y añadir objetivos de precio
        if analysis['signal'] not in ['NEUTRAL', 'ERROR'] and not pd.isna(ind['Close_LTF']) and not pd.isna(ind['ATR_LTF']):
            analysis['price_targets'] = self._calculate_price_targets_stops(
                analysis['signal'], ind['Close_LTF'], 
                analysis['fib_levels'], ind['ATR_LTF'] # df_ltf e indicators no son necesarios si pasamos ATR y Fibs
            )
            pt = analysis['price_targets']
            analysis['reasoning'].append(f"Targets(ATR/Fib): TP1={self._s(pt['tp1'])}, SL={self._s(pt['sl'])}")
        
        # Limpiar duplicados y asegurar que la decisión final esté al principio
        temp_reasoning = []
        seen_reasons = set()
        decision_reason = f"DECISIÓN CERVELLO: {analysis['signal']} (Conf: {analysis['confidence']:.0%})"
        temp_reasoning.append(decision_reason); seen_reasons.add(decision_reason)
        for r in analysis['reasoning']:
            if r not in seen_reasons and not r.startswith("DECISIÓN CERVELLO:"):
                temp_reasoning.append(r); seen_reasons.add(r)
        analysis['reasoning'] = temp_reasoning[:15]

        logger.info(
            f"Cervello {pair_symbol}: {analysis['signal']} ({analysis['confidence']:.0%}) | "
            f"TP1:{self._s(analysis['price_targets']['tp1'])} SL:{self._s(analysis['price_targets']['sl'])} | "
            f"MktProf: {market_profile.get('overall_sentiment','N/A')}/{market_profile.get('regime','N/A')}"
        )
        return analysis