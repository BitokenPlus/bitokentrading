# exchange_client.py (Versión con corrección para obtención de OI)
import ccxt
import pandas as pd
import time
from utils import get_logger 
import config # Importa tu archivo de configuración completo
import math 
import numpy as np # Para np.nan
import logging # Para acceder a logger.level

logger = get_logger(__name__) 

class ExchangeClient:
    def __init__(self):
        logger.info(f"Inicializando ExchangeClient para Binance ({config.MARKET_TYPE.upper()}). Version: {config.APP_VERSION}")
        exchange_params = {
            'enableRateLimit': True,
            'options': {
                'defaultType': config.MARKET_TYPE, 
                'adjustForTimeDifference': True, 
            }
        }
        
        binance_api_key = getattr(config, 'BINANCE_API_KEY', "")
        binance_api_secret = getattr(config, 'BINANCE_API_SECRET', "")

        if binance_api_key and binance_api_secret:
            exchange_params['apiKey'] = binance_api_key
            exchange_params['secret'] = binance_api_secret
            logger.info("Usando API keys de config.py.")
        else:
            logger.info("NO se están usando API keys (acceso público).")

        try:
            self.exchange = ccxt.binance(exchange_params)
            logger.info("Cargando mercados desde Binance...")
            self.exchange.load_markets(reload=True) 
            if self.exchange.markets:
                logger.info(f"Mercados cargados exitosamente: {len(self.exchange.markets)} encontrados.")
            else:
                logger.error("Fallo al cargar mercados o la lista de mercados está vacía.")
                self.exchange = None 
                raise ccxt.ExchangeError("No se pudieron cargar los mercados.")
        except Exception as e:
            logger.error(f"Error crítico al inicializar CCXT o cargar mercados: {e}", exc_info=True)
            self.exchange = None
            raise 

    def get_all_valid_future_pairs_data(self):
        """
        Identifica futuros perpetuos USDT-M y obtiene sus datos de ticker (volumen, precio).
        Open Interest se inicializa a np.nan para ser llenado después.
        """
        if not self.exchange or not self.exchange.markets:
            logger.error("Exchange no inicializado o mercados no cargados en get_all_valid_future_pairs_data.")
            return []

        target_quote = config.TARGET_QUOTE_ASSET.upper()
        all_markets = self.exchange.markets
        potential_symbols = []
        
        logger.debug(f"Filtrando {len(all_markets)} mercados para futuros {target_quote}-M activos...")
        for symbol, market_details in all_markets.items():
            if not isinstance(market_details, dict): continue

            is_swap = market_details.get('swap', False)
            quote_asset_val = market_details.get('quote')
            settle_asset_val = market_details.get('settle')
            has_correct_quote = quote_asset_val.upper() == target_quote if quote_asset_val else False
            has_correct_settle = settle_asset_val.upper() == target_quote if settle_asset_val else False
            is_active = market_details.get('active', False) 
            contract_type = market_details.get('info', {}).get('contractType')
            status = market_details.get('info', {}).get('status')

            if logger.level == logging.DEBUG: # Log detallado solo en DEBUG
                logger.debug(f"Evaluando {symbol}: swap={is_swap}, q={quote_asset_val}({has_correct_quote}), "
                             f"s={settle_asset_val}({has_correct_settle}), act={is_active}, "
                             f"cType={contract_type}, stat={status}")

            if (is_swap and has_correct_quote and has_correct_settle and is_active and
                contract_type == 'PERPETUAL' and status == 'TRADING'):
                potential_symbols.append(symbol)
        
        if not potential_symbols:
            logger.error(f"¡FALLO CRÍTICO! No se encontraron símbolos que cumplan los criterios básicos de ser futuros perpetuos {target_quote}-M activos y en trading.")
            return []
        logger.info(f"Se encontraron {len(potential_symbols)} símbolos que parecen ser futuros {target_quote}-M válidos (Paso 1 completado).")
        logger.debug(f"Primeros 10 símbolos potenciales (Paso 1): {potential_symbols[:10]}")

        # Paso 2: Obtener tickers (volumen, precio) para estos símbolos
        all_pairs_initial_data = []
        chunk_size = getattr(config, 'TICKER_CHUNK_SIZE', 50)
        num_chunks = (len(potential_symbols) + chunk_size - 1) // chunk_size
        
        logger.info(f"Intentando obtener tickers para {len(potential_symbols)} símbolos en {num_chunks} chunks...")
        for i in range(num_chunks):
            chunk = potential_symbols[i*chunk_size : (i+1)*chunk_size]
            logger.debug(f"Procesando chunk de tickers {i+1}/{num_chunks} ({len(chunk)} símbolos)...")
            try:
                if not chunk: continue
                tickers = self.exchange.fetch_tickers(symbols=chunk)
                
                for symbol_ticker, ticker_data in tickers.items():
                    if not isinstance(ticker_data, dict): continue
                    vol_quote = ticker_data.get('quoteVolume', 0.0)
                    last_price = ticker_data.get('last', 0.0)
                    
                    all_pairs_initial_data.append({
                        'symbol': symbol_ticker,
                        'volume_24h_usd': vol_quote if not math.isnan(vol_quote) else 0.0,
                        'open_interest_usd': np.nan, # Dejar como NaN, se llenará después
                        'last_price': last_price if not math.isnan(last_price) else 0.0,
                    })
                if num_chunks > 1 and i < num_chunks -1: time.sleep(0.2) 
            except Exception as e:
                logger.error(f"Error obteniendo tickers para chunk {i+1}: {e}", exc_info=True)
        
        if not all_pairs_initial_data:
             logger.warning("No se pudieron obtener datos de tickers para ninguno de los símbolos potenciales.")
             return [] # Si no hay tickers, no podemos continuar

        logger.info(f"Datos iniciales de tickers (Vol/Precio) obtenidos para {len(all_pairs_initial_data)} pares (Paso 2 completado).")
        return all_pairs_initial_data

    def get_target_quote_asset_pairs(self):
        """
        Función principal para obtener la lista de pares a analizar.
        Obtiene OI individualmente después de un pre-filtro por volumen.
        Aplica filtros de debug muy bajos.
        """
        logger.info("Ejecutando get_target_quote_asset_pairs (con obtención individual de OI)...")
        
        specific_pairs_list = getattr(config, 'SPECIFIC_PAIRS_TO_MONITOR', [])
        if not isinstance(specific_pairs_list, list): specific_pairs_list = []

        if specific_pairs_list: 
            logger.info(f"Modo Selección: Usando lista específica: {specific_pairs_list}")
            # Aquí podríamos validar si estos pares existen y son válidos
            # pero para simplificar el debug, los pasamos directamente.
            return specific_pairs_list

        # Paso 1 y 2 (combinados en get_all_valid_future_pairs_data): Obtener símbolos y datos de ticker (Vol/Precio)
        initial_pairs_data = self.get_all_valid_future_pairs_data()
        if not initial_pairs_data:
            logger.error("get_all_valid_future_pairs_data no devolvió datos. No hay pares para analizar.")
            return []

        # Paso 3: Pre-filtrar por volumen ANTES de obtener OI para reducir llamadas API
        # Usar un valor de config para este pre-filtro, o un default razonable
        pre_filter_min_vol = getattr(config, 'PRE_OI_FETCH_MIN_VOLUME', 1_000_000) # Ej: 1M USD
        
        pairs_needing_oi = [
            p for p in initial_pairs_data 
            if p.get('volume_24h_usd', 0.0) >= pre_filter_min_vol
        ]
        logger.info(f"Después de pre-filtro por volumen (> {pre_filter_min_vol:,.0f}), "
                    f"se obtendrá OI para {len(pairs_needing_oi)} de {len(initial_pairs_data)} pares.")

        if not pairs_needing_oi:
            logger.warning(f"Ningún par pasó el pre-filtro de volumen ({pre_filter_min_vol:,.0f}). No se obtendrá OI.")
            # Decidir si devolver la lista 'initial_pairs_data' sin OI, o una lista vacía.
            # Por ahora, si no podemos obtener OI para un subconjunto razonable, devolvemos vacío.
            return [] 

        # Paso 4: Obtener Open Interest individualmente para los pares pre-filtrados
        logger.info(f"Paso 4: Obteniendo Open Interest para {len(pairs_needing_oi)} pares (esto puede tardar)...")
        all_data_complete_with_oi = [] # Lista para guardar los datos con OI
        processed_oi_count = 0
        oi_fetch_errors = 0

        for pair_info in pairs_needing_oi:
            symbol = pair_info['symbol']
            logger.debug(f"Obteniendo OI para {symbol} ({processed_oi_count+1}/{len(pairs_needing_oi)})...")
            
            oi_api_response = self.fetch_open_interest(symbol) # Llama al método fetch_open_interest
            
            current_pair_oi_usd = 0.0 # Default a 0.0 si no se puede obtener/calcular
            if oi_api_response:
                # Intentar obtener 'openInterestValue' (en USDT)
                oi_val_direct = oi_api_response.get('openInterestValue')
                if oi_val_direct is not None and not math.isnan(oi_val_direct):
                    current_pair_oi_usd = oi_val_direct
                else: # Si no está, intentar calcular desde 'openInterestAmount'
                    oi_amount_base = oi_api_response.get('openInterestAmount')
                    last_known_price = pair_info.get('last_price', 0.0) # Usar precio del ticker
                    
                    if oi_amount_base is not None and not math.isnan(oi_amount_base) and last_known_price > 0:
                        # Obtener contractSize del market object para precisión
                        market_object = self.exchange.market(symbol)
                        contract_s = market_object.get('contractSize', 1.0) if market_object else 1.0
                        current_pair_oi_usd = oi_amount_base * last_known_price * contract_s
                    # else: # Se mantiene 0.0 si no se puede calcular
            else:
                oi_fetch_errors +=1
                logger.debug(f"No se obtuvieron datos de OI para {symbol} desde fetch_open_interest.")
            
            # Actualizar/añadir el OI al diccionario del par
            pair_info['open_interest_usd'] = current_pair_oi_usd if not math.isnan(current_pair_oi_usd) else 0.0
            all_data_complete_with_oi.append(pair_info) # Añadir a la lista final
            
            processed_oi_count += 1
            # Pausa periódica para no saturar la API
            if processed_oi_count % getattr(config, 'OI_FETCH_BATCH_SIZE', 20) == 0 and processed_oi_count < len(pairs_needing_oi):
                pause_duration = getattr(config, 'OI_FETCH_BATCH_PAUSE_S', 0.5)
                logger.debug(f"Pausa de {pause_duration}s después de {processed_oi_count} llamadas de OI.")
                time.sleep(pause_duration)

        logger.info(f"Paso 4 completado: OI obtenido para {len(all_data_complete_with_oi)} pares. Errores de fetch OI: {oi_fetch_errors}.")
        if not all_data_complete_with_oi:
            logger.error("No se pudo obtener y procesar OI para ningún par después del pre-filtro de volumen.")
            return []

        # Paso 5: Aplicar filtros de depuración MUY BAJOS a los datos completos
        # En una versión de producción, aquí iría la lógica de los modos de selección y ajuste automático.
        debug_min_vol_final = 1.0 
        debug_min_oi_final = 1.0 # Ahora este filtro tiene OI real para comparar
        top_n_final_debug = getattr(config, 'ADAPTIVE_TOP_N', 200) # Tomar un número grande

        logger.info(f"Paso 5: Aplicando filtros de depuración finales: Vol > {debug_min_vol_final}, OI > {debug_min_oi_final} sobre {len(all_data_complete_with_oi)} pares.")
        
        final_filtered_list_debug = [
            p for p in all_data_complete_with_oi
            if p.get('volume_24h_usd', 0.0) >= debug_min_vol_final and \
               p.get('open_interest_usd', 0.0) >= debug_min_oi_final
        ]
        
        # Log para ver los datos de OI después de la obtención individual
        if logger.level == logging.DEBUG and all_data_complete_with_oi:
            logger.debug("Primeros 5 pares en 'all_data_complete_with_oi' (CON OI OBTENIDO):")
            for idx, p_debug_oi in enumerate(all_data_complete_with_oi[:5]):
                 logger.debug(f"  Par: {p_debug_oi.get('symbol')}, "
                              f"Vol: {p_debug_oi.get('volume_24h_usd', 0):,.0f}, "
                              f"OI: {p_debug_oi.get('open_interest_usd', 0):,.0f}")


        if not final_filtered_list_debug:
            logger.warning(f"¡FILTRO FINAL DE DEBUG FALLÓ! NINGÚN PAR FUE SELECCIONADO DESPUÉS DE OBTENER OI. "
                           f"Esto sugiere que el OI sigue siendo cero/bajo para todos los pares procesados, o un problema en el filtro.")
        else:
            # Ordenar por volumen (o cualquier cosa) y tomar el top N
            final_filtered_list_debug.sort(key=lambda x: x.get('volume_24h_usd', 0.0), reverse=True)
        
        final_selected_symbols_debug = [p['symbol'] for p in final_filtered_list_debug[:top_n_final_debug]]

        logger.info(f"Pares seleccionados con filtros de debug ({len(final_selected_symbols_debug)}): {final_selected_symbols_debug[:20]}")
        return final_selected_symbols_debug

    # --- fetch_ohlcv, fetch_open_interest, fetch_funding_rate ---
    def fetch_ohlcv(self, symbol, timeframe, limit):
        if not self.exchange: return None
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, params={'price': 'mark'})
            if not ohlcv: 
                # logger.warning(f"fetch_ohlcv devolvió lista vacía para {symbol} TF={timeframe}.")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            return df
        except ccxt.BadSymbol: 
            logger.debug(f"Símbolo '{symbol}' inválido para fetch_ohlcv.")
            return None
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} ({timeframe}): {e}", exc_info=False)
            return None

    def fetch_open_interest(self, symbol):
        if not self.exchange or config.MARKET_TYPE != 'future': return None
        try:
            if self.exchange.has['fetchOpenInterest']:
                oi_data = self.exchange.fetch_open_interest(symbol)
                return oi_data
            return None
        except ccxt.BadSymbol: 
            logger.debug(f"Símbolo '{symbol}' inválido para fetch_open_interest.")
            return None
        except Exception as e:
            # logger.error(f"Error fetching Open Interest for {symbol}: {e}", exc_info=False) # Silenciado para reducir ruido
            return None

    def fetch_funding_rate(self, symbol):
        if not self.exchange or config.MARKET_TYPE != 'future': return None
        try:
            market = self.exchange.market(symbol) 
            if not market or not market.get('swap', False): return None
            if self.exchange.has['fetchFundingRate']:
                fr_data = self.exchange.fetch_funding_rate(symbol)
                return fr_data
            return None
        except ccxt.BadSymbol: 
            logger.debug(f"Símbolo '{symbol}' inválido para fetch_funding_rate.")
            return None
        except Exception as e:
            # logger.error(f"Error fetching Funding Rate for {symbol}: {e}", exc_info=False) # Silenciado
            return None