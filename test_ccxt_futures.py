# test_ccxt_futures.py (CON CORRECCIÓN PARA AttributeError)
import ccxt
import json # Para pretty print

print(f"CCXT Version: {ccxt.__version__}")

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
    }
})

try:
    print("Cargando mercados de Binance Futures (sin API keys)...")
    markets = exchange.load_markets()
    print(f"Total de mercados cargados: {len(markets)}")

    TARGET_QUOTE_ASSET_PARA_FILTRAR = "USDT"
    future_usdt_m_pairs = []
    count = 0

    print(f"\nBuscando Futuros Perpetuos {TARGET_QUOTE_ASSET_PARA_FILTRAR}-M:")
    for symbol, market_details in markets.items():
        if not isinstance(market_details, dict): # Seguridad extra
            # print(f"Saltando {symbol}, market_details no es un diccionario: {type(market_details)}")
            continue

        is_swap = market_details.get('swap', False)
        
        # --- CORRECCIÓN AQUÍ ---
        quote_asset = market_details.get('quote')
        settle_asset = market_details.get('settle')

        quote_is_target = False
        if quote_asset is not None:
            quote_is_target = quote_asset.upper() == TARGET_QUOTE_ASSET_PARA_FILTRAR
        # else:
            # print(f"Saltando {symbol} debido a quote_asset None. Detalles: {market_details}")


        settle_is_target = False
        if settle_asset is not None:
            settle_is_target = settle_asset.upper() == TARGET_QUOTE_ASSET_PARA_FILTRAR
        # else:
            # print(f"Saltando {symbol} debido a settle_asset None. Detalles: {market_details}")

        is_active_trading = market_details.get('active', True)
        contract_type_info = market_details.get('info', {}).get('contractType')
        status_info = market_details.get('info', {}).get('status')

        if (is_swap and
            quote_is_target and # Ahora es un booleano directamente
            settle_is_target and # Ahora es un booleano directamente
            is_active_trading and
            (contract_type_info == 'PERPETUAL' if contract_type_info else True) and
            (status_info == 'TRADING' if status_info else True)):

            future_usdt_m_pairs.append(symbol)
            if count < 10:
                print("-" * 30)
                print(f"Símbolo CCXT: {symbol}")
                print(f"  Tipo CCXT: {market_details.get('type')}")
                print(f"  Activo: {market_details.get('active')}")
                print(f"  Base: {market_details.get('base')}, Quote: {quote_asset}, Settle: {settle_asset}") # Mostrar original
                print(f"  Swap: {market_details.get('swap')}, Linear: {market_details.get('linear')}")
                print(f"  Info['contractType']: {contract_type_info}")
                print(f"  Info['status']: {status_info}")
            count += 1
        # elif is_swap and (quote_asset is None or settle_asset is None): # Opcional: Loguear los que se saltan por esto
            # print(f"DEBUG: {symbol} es swap pero quote o settle es None. Quote: {quote_asset}, Settle: {settle_asset}")


    print("-" * 50)
    print(f"Total de pares de Futuros Perpetuos {TARGET_QUOTE_ASSET_PARA_FILTRAR}-M encontrados: {len(future_usdt_m_pairs)}")
    if future_usdt_m_pairs:
        print(f"Primeros 10: {future_usdt_m_pairs[:10]}")

    # ... (resto del script para probar fetch_tickers y fetch_ohlcv sin cambios) ...
    # Prueba fetch_tickers para algunos de estos
    if len(future_usdt_m_pairs) >= 5:
        test_symbols_for_tickers = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT']
        valid_test_symbols = [s for s in test_symbols_for_tickers if s in future_usdt_m_pairs]

        if valid_test_symbols:
            print(f"\nProbando fetch_tickers para: {valid_test_symbols} (sin API keys)...")
            try:
                tickers = exchange.fetch_tickers(symbols=valid_test_symbols)
                for sym, tick_data in tickers.items():
                    print(f"Ticker para {sym}: Precio={tick_data.get('last')}, VolQuote={tick_data.get('quoteVolume')}, OIVal={tick_data.get('openInterestValue')}")
            except Exception as e_tick:
                print(f"Error en fetch_tickers: {e_tick}")
        else:
            print(f"\nNo se pudieron seleccionar símbolos válidos de {test_symbols_for_tickers} en la lista filtrada ({len(future_usdt_m_pairs)} pares) para probar fetch_tickers.")
            if future_usdt_m_pairs and len(future_usdt_m_pairs) >= 5: # Probar con los primeros de la lista si los predefinidos no están
                valid_test_symbols = future_usdt_m_pairs[:5]
                print(f"Intentando con los primeros 5 de la lista filtrada: {valid_test_symbols}")
                try:
                    tickers = exchange.fetch_tickers(symbols=valid_test_symbols)
                    for sym, tick_data in tickers.items():
                        print(f"Ticker para {sym}: Precio={tick_data.get('last')}, VolQuote={tick_data.get('quoteVolume')}, OIVal={tick_data.get('openInterestValue')}")
                except Exception as e_tick:
                    print(f"Error en fetch_tickers con los primeros 5: {e_tick}")

            
    # Prueba fetch_ohlcv para un par
    if future_usdt_m_pairs:
        test_ohlcv_symbol = 'BTC/USDT:USDT' 
        if test_ohlcv_symbol not in future_usdt_m_pairs: # Si BTC/USDT:USDT no está, tomar el primero
            test_ohlcv_symbol = future_usdt_m_pairs[0]
            
        print(f"\nProbando fetch_ohlcv para {test_ohlcv_symbol} (timeframe 1h, limit 5) (sin API keys)...")
        try:
            ohlcv = exchange.fetch_ohlcv(test_ohlcv_symbol, timeframe='1h', limit=5)
            if ohlcv:
                print(f"OHLCV para {test_ohlcv_symbol}:")
                for candle in ohlcv:
                    print(f"  Timestamp: {exchange.iso8601(candle[0])}, O: {candle[1]}, H: {candle[2]}, L: {candle[3]}, C: {candle[4]}, V: {candle[5]}")
            else:
                print("No se recibieron datos OHLCV.")
        except Exception as e_ohlcv:
            print(f"Error en fetch_ohlcv: {e_ohlcv}")


except ccxt.NetworkError as e:
    print(f"Error de Red CCXT: {type(e).__name__}, {e.args}")
except ccxt.ExchangeError as e:
    print(f"Error de Exchange CCXT: {type(e).__name__}, {e.args}")
except Exception as e:
    print(f"Error General: {type(e).__name__} {e.args}") # Modificado para imprimir el tipo de error también
    import traceback
    traceback.print_exc() # Imprimir el traceback completo para más detalles
