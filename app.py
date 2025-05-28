# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import math 
import numpy as np

from utils import get_logger # Para logging
import config # Configuraciones globales
from exchange_client import ExchangeClient # Para interactuar con el exchange
from indicators import IndicatorCalculator # Para calcular indicadores técnicos
from market_profile import MarketProfileAnalyzer # NUEVO: Para analizar el perfil del mercado global
from strategy_cervello import CervelloStrategy # NUEVO: Estrategia de IA "Cervello"

logger = get_logger(__name__) # Logger específico para app.py

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title=f"FuturosIAPro🧠 BitokenPlus🐘({config.APP_VERSION})", # Usa APP_VERSION de config
    page_icon="🐘", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Estilo CSS Personalizado ---
# (Asegúrate de tener tu bloque CSS completo aquí si lo personalizaste)
st.markdown("""
<style>
    /* Estilos Generales */
    .stApp {
        background-color: #0E1117; color: #FAFAFA;
    }
    h1, h2, h3, h4, h5, h6 { color: #FF9900; }
    /* Botones */
    .stButton>button {
        background-color: #FF9900; color: #0E1117; border: none;
        padding: 0.5rem 1rem; border-radius: 0.25rem; font-weight: bold;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover { background-color: #FFAA33; color: #000000; }
    /* Sidebar */
    .st-emotion-cache-16txtl3 { background-color: #1A1F2E; }
    .st-emotion-cache-16txtl3 .st-emotion-cache-16idsys p,
    .st-emotion-cache-16txtl3 .st-emotion-cache-16idsys li { color: #FAFAFA !important; }
    .st-emotion-cache-16txtl3 .stSelectbox label,
    .st-emotion-cache-16txtl3 .stButton>label,
    .st-emotion-cache-16txtl3 .stTextInput label,
    .st-emotion-cache-16txtl3 .stNumberInput label { color: #FF9900 !important; font-weight: bold; }
    /* Expanders */
    .stExpander { border: 1px solid #FF9900; border-radius: 0.25rem; background-color: #1A1F2E; margin-bottom: 1rem; }
    .stExpander header { color: #FF9900; font-weight: bold; background-color: #1A1F2E; }
    /* Métricas */
    .stMetric { background-color: #1A1F2E; padding: 1rem; border-radius: 0.25rem; border: 1px solid #444444; text-align: center; }
    .stMetric label { color: #AAAAAA; font-size: 0.9em; }
    .stMetric value { color: #FAFAFA; font-size: 1.5em; font-weight: bold; }
    /* Dataframes */
    .stDataFrame th { background-color: #FF9900; color: #0E1117; font-weight: bold; }
    /* Clases de Señal */
    .signal-long { color: #2ECC71 !important; font-weight: bold; }
    .signal-short { color: #E74C3C !important; font-weight: bold; }
    .signal-cautious_long { color: #F1C40F !important; font-weight: bold; }
    .signal-cautious_short { color: #E67E22 !important; font-weight: bold; }
    .signal-neutral { color: #95A5A6 !important; }
    /* Spinner */
    .stSpinner > div > div { border-top-color: #FF9900 !important; border-right-color: #FF9900 !important; }
</style>
""", unsafe_allow_html=True)

# --- Estado de Sesión de Streamlit ---
# (Usar st.session_state para persistir datos entre interacciones)
if 'selected_ltf' not in st.session_state: st.session_state.selected_ltf = config.LTF
if 'pairs_to_operate' not in st.session_state: st.session_state.pairs_to_operate = []
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'selected_details_pair' not in st.session_state: st.session_state.selected_details_pair = None
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None
if 'global_market_profile_cache' not in st.session_state: st.session_state.global_market_profile_cache = None

# --- Inicialización de Componentes Backend (Cacheado) ---
@st.cache_resource # Cachear para mejorar rendimiento en recargas
def init_backend_components():
    logger.info(f"Inicializando componentes backend ({config.APP_VERSION})...")
    client_instance, calculator_instance, market_profiler_instance, strategy_instance = None, None, None, None
    initialization_error = None
    try:
        client_instance = ExchangeClient()
        calculator_instance = IndicatorCalculator()
        market_profiler_instance = MarketProfileAnalyzer(calculator_instance) # Pasa la instancia de calculator
        strategy_instance = CervelloStrategy() # Usa config para pesos y parámetros
        logger.info("Todos los componentes backend inicializados correctamente.")
    except Exception as e:
        initialization_error = f"Error Crítico al inicializar componentes backend: {e}"
        logger.error(initialization_error, exc_info=True)
    return client_instance, calculator_instance, market_profiler_instance, strategy_instance, initialization_error

# Obtener instancias cacheadas o None si falla la inicialización
client, calculator, market_profiler, cervello_strategy, init_error = init_backend_components()

# --- Título Principal y Estado de Conexión ---
st.title(f"FuturosIAPro BitokenPlus 🐘{config.APP_VERSION}")
if init_error:
    st.error(f"La aplicación no pudo iniciarse: {init_error}")
    st.stop() # Detener la ejecución de la app si los componentes críticos fallan

status_col1, status_col2 = st.columns([1, 3])
with status_col1:
    connection_ok = client and client.exchange and hasattr(client.exchange, 'rateLimit') # Chequeo más robusto
    connection_status_text = "🟢 Conectado" if connection_ok else "🔴 Desconectado"
    st.caption(f"**Estado Conexión ({config.MARKET_TYPE.upper()}):** {connection_status_text}")
with status_col2:
    last_scan_time_str = st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S %Z') if st.session_state.last_scan_time else 'Nunca'
    st.caption(f"**Último Análisis:** {last_scan_time_str}")

# --- Sidebar para Controles de Usuario ---
with st.sidebar:
    st.header("🧠 Configuración del Análisis IA")
    
    # Selección de Temporalidades
    tf_options = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'] # Opciones LTF
    # Usar st.session_state para la temporalidad seleccionada para que persista
    st.session_state.selected_ltf = st.selectbox(
        "Temporalidad Principal (LTF):", tf_options, 
        index=tf_options.index(st.session_state.selected_ltf) if st.session_state.selected_ltf in tf_options else tf_options.index(config.LTF)
    )
    derived_htf = config.HTF_DERIVATION_MAP.get(st.session_state.selected_ltf, '4h') # Derivar HTF
    st.caption(f"Temporalidad de Confirmación (HTF) usada: **{derived_htf}**")

    # Modo de Selección de Pares
    pair_selection_options_map = {
        "ADAPTIVE_FOCUS": "Foco Adaptativo (Recomendado)",
        "TOP_BY_LIQUIDITY": "Top por Liquidez (Estricto)",
        "USER_DEFINED_FILTERS": "Filtros Manuales (Vol/OI)",
        "SPECIFIC_LIST": "Lista Específica (de config.py)"
    }
    default_pair_selection_mode = getattr(config, 'PAIR_SELECTION_MODE', "ADAPTIVE_FOCUS")

    # Forzar "Lista Específica" si config.SPECIFIC_PAIRS_TO_MONITOR está poblado
    specific_pairs_from_config = getattr(config, 'SPECIFIC_PAIRS_TO_MONITOR', [])
    if not isinstance(specific_pairs_from_config, list): specific_pairs_from_config = []

    if specific_pairs_from_config: # Si hay pares específicos en config
        st.selectbox("Modo Selección Pares:", [pair_selection_options_map["SPECIFIC_LIST"]], 0, disabled=True,
                     help="Definido en config.py: SPECIFIC_PAIRS_TO_MONITOR")
        # Actualizar config.PAIR_SELECTION_MODE en la sesión actual si es necesario
        # (aunque ExchangeClient lo leerá directamente de config)
        st.caption(f"Usando lista de {len(specific_pairs_from_config)} pares de `config.py`.")
        current_pair_selection_mode_key = "SPECIFIC_LIST"
    else:
        selected_pair_mode_display = st.selectbox(
            "Modo Selección Pares:",
            options=list(pair_selection_options_map.values()),
            index=list(pair_selection_options_map.keys()).index(default_pair_selection_mode)
        )
        current_pair_selection_mode_key = [k for k, v in pair_selection_options_map.items() if v == selected_pair_mode_display][0]
    
    # Actualizar config.PAIR_SELECTION_MODE dinámicamente para que ExchangeClient lo use
    # Esto es un poco un hack, idealmente se pasaría como parámetro, pero para simplificar:
    config.PAIR_SELECTION_MODE = current_pair_selection_mode_key

    # Mostrar filtros manuales solo si ese modo está seleccionado
    if config.PAIR_SELECTION_MODE == "USER_DEFINED_FILTERS":
        st.caption("Filtros Manuales (Vol/OI):")
        # Usar valores de config como default y actualizarlos
        config.USER_MIN_VOLUME = st.number_input(f"Vol Min 24h ({config.TARGET_QUOTE_ASSET})", 
                                                 value=config.USER_MIN_VOLUME, step=1_000_000, min_value=0)
        if config.MARKET_TYPE == 'future':
            config.USER_MIN_OI = st.number_input(f"OI Mínimo ({config.TARGET_QUOTE_ASSET})", 
                                                  value=config.USER_MIN_OI, step=500_000, min_value=0)
    # Añadir captions informativas para otros modos
    elif config.PAIR_SELECTION_MODE == "ADAPTIVE_FOCUS":
         st.caption(f"Analizará los TOP {getattr(config, 'ADAPTIVE_TOP_N', 30)} pares (ajuste automático).")
    elif config.PAIR_SELECTION_MODE == "TOP_BY_LIQUIDITY":
         st.caption(f"Analizará los TOP {getattr(config, 'TOP_N_LIQUID', 30)} pares más líquidos.")

    # Botón de Análisis
    analyze_button = st.button("🚀 Analizar Mercado (IA Cervello)")
    st.markdown("---")

    # Pares en Seguimiento (sección para añadir/quitar manualmente)
    st.header("👀 Pares en Seguimiento")
    if st.session_state.pairs_to_operate:
        selected_for_removal = st.multiselect(
            "Quitar de seguimiento:", 
            options=st.session_state.pairs_to_operate, 
            default=[],
            label_visibility="collapsed"
        )
        if selected_for_removal:
            st.session_state.pairs_to_operate = [p for p in st.session_state.pairs_to_operate if p not in selected_for_removal]
            st.rerun() # Recargar para actualizar la lista
        
        # Mostrar los que quedan
        if st.session_state.pairs_to_operate:
            st.expander("Ver seguimiento activo", expanded=False).write(st.session_state.pairs_to_operate)

    else:
        st.write("Añade pares desde la tabla de resultados.")
    st.info("La 'IA Cervello' aplica lógica avanzada y perfil de mercado. Revisa `strategy_cervello 🧠`")

# --- Lógica Principal de Escaneo ---
if analyze_button:
    # Validar que todos los componentes estén listos
    if not all([client, calculator, market_profiler, cervello_strategy]):
        st.error("Error crítico: Uno o más componentes del backend no pudieron inicializarse. Revisa los logs.")
        st.stop()

    current_ltf = st.session_state.selected_ltf # Usar la temporalidad seleccionada
    current_htf = config.HTF_DERIVATION_MAP.get(current_ltf, '4h')

    st.session_state.analysis_results[current_ltf] = [] # Limpiar resultados anteriores para esta TF
    st.session_state.selected_details_pair = None # Resetear la selección de detalles

    # Fase 1: Obtener y Analizar Perfil del Mercado Global
    with st.spinner(f"Fase 1: Analizando Perfil del Mercado Global ({config.GLOBAL_MARKET_SYMBOL} en {current_htf})..."):
        df_global_market_raw = client.fetch_ohlcv(config.GLOBAL_MARKET_SYMBOL, current_htf, config.DATA_FETCH_LIMIT_HTF)
        
        if df_global_market_raw is not None and not df_global_market_raw.empty:
            df_global_market_indicators = calculator.add_indicators(df_global_market_raw.copy())
            if df_global_market_indicators is not None and not df_global_market_indicators.empty:
                st.session_state.global_market_profile_cache = market_profiler.get_global_market_profile(df_global_market_indicators)
                profile_info = st.session_state.global_market_profile_cache
                st.success(f"Perfil Mercado: {profile_info.get('regime','N/A')}, Sent: {profile_info.get('overall_sentiment','N/A')}, Vol: {profile_info.get('volatility_level','N/A')}")
            else:
                st.warning("No se pudieron calcular indicadores para el perfil del mercado global. Usando perfil por defecto.")
                st.session_state.global_market_profile_cache = market_profiler.get_global_market_profile(None) # Obtener perfil por defecto
        else:
            st.error(f"No se pudieron obtener datos para {config.GLOBAL_MARKET_SYMBOL}. Usando perfil de mercado por defecto.")
            st.session_state.global_market_profile_cache = market_profiler.get_global_market_profile(None)

    # Fase 2: Seleccionar y Analizar Pares Individuales
    global_profile_to_use = st.session_state.global_market_profile_cache # Usar el perfil cacheado
    
    with st.spinner(f"Fase 2: Seleccionando y analizando pares ({current_ltf})..."):
        logger.info(f"SCAN INICIADO: LTF={current_ltf}, HTF={current_htf}, Modo Selección={config.PAIR_SELECTION_MODE}")
        
        pairs_to_scan = client.get_target_quote_asset_pairs() # Usa la lógica de selección interna
        
        if not pairs_to_scan:
            st.warning(f"No se encontraron pares de futuros {config.TARGET_QUOTE_ASSET} con los criterios del modo '{config.PAIR_SELECTION_MODE}'. Revisa la configuración o los logs detallados (DEBUG).")
            st.stop()
        
        st.info(f"Se analizarán {len(pairs_to_scan)} pares. Perfil Global Aplicado: {global_profile_to_use.get('regime','N/A')}")
        scan_progress_bar = st.progress(0, text="Iniciando análisis de pares...")
        processed_results_list = []

        for i, pair_symbol_to_scan in enumerate(pairs_to_scan):
            progress_msg = f"Procesando {pair_symbol_to_scan} ({i+1}/{len(pairs_to_scan)})..."
            scan_progress_bar.progress((i + 1) / len(pairs_to_scan), text=progress_msg)
            
            pair_analysis_data = {'pair': pair_symbol_to_scan, 'error': None} # Init para este par
            try:
                # Obtener datos OHLCV
                df_ltf_pair_raw = client.fetch_ohlcv(pair_symbol_to_scan, current_ltf, config.DATA_FETCH_LIMIT_LTF)
                time.sleep(0.025) # Pausa mínima
                df_htf_pair_raw = client.fetch_ohlcv(pair_symbol_to_scan, current_htf, config.DATA_FETCH_LIMIT_HTF)
                time.sleep(0.025)

                if df_ltf_pair_raw is None or df_htf_pair_raw is None or df_ltf_pair_raw.empty or df_htf_pair_raw.empty:
                    pair_analysis_data['error'] = "Datos OHLCV insuficientes para el par."
                    processed_results_list.append(pair_analysis_data); continue

                # Calcular Indicadores
                df_ltf_pair_with_indicators = calculator.add_indicators(df_ltf_pair_raw.copy())
                df_htf_pair_with_indicators = calculator.add_indicators(df_htf_pair_raw.copy())
                
                if df_ltf_pair_with_indicators is None or df_htf_pair_with_indicators is None:
                    pair_analysis_data['error'] = "Fallo al calcular indicadores para el par."
                    processed_results_list.append(pair_analysis_data); continue

                # Obtener datos de Futuros (OI, FR)
                oi_pair_data, fr_pair_data = None, None
                if config.MARKET_TYPE == 'future':
                    oi_pair_data = client.fetch_open_interest(pair_symbol_to_scan); time.sleep(0.015)
                    fr_pair_data = client.fetch_funding_rate(pair_symbol_to_scan); time.sleep(0.015)
                
                # Analizar con CervelloStrategy
                cervello_output = cervello_strategy.analyze_pair(
                    df_ltf_pair_with_indicators, df_htf_pair_with_indicators, 
                    pair_symbol_to_scan, 
                    global_profile_to_use, # Pasar el perfil del mercado global
                    oi_pair_data, fr_pair_data
                )
                processed_results_list.append(cervello_output)

            except Exception as e_scan_pair:
                logger.error(f"Error crítico procesando el par {pair_symbol_to_scan}: {e_scan_pair}", exc_info=True)
                pair_analysis_data['error'] = f"Excepción Inesperada: {str(e_scan_pair)[:120]}" # Mensaje corto para UI
                processed_results_list.append(pair_analysis_data)
        
        st.session_state.analysis_results[current_ltf] = processed_results_list
        st.session_state.last_scan_time = pd.Timestamp.now(tz='UTC')
        scan_progress_bar.progress(1.0, "Análisis de todos los pares completado.")
        st.success(f"Análisis Cervello de {len(pairs_to_scan)} pares finalizado para {current_ltf}.")
        st.rerun() # Recargar para mostrar resultados


# --- Área Principal de Resultados ---
active_ltf_display = st.session_state.selected_ltf # Usar la TF seleccionada para mostrar resultados
st.header(f"Resultados del Análisis IA ({active_ltf_display})")

results_for_display = st.session_state.analysis_results.get(active_ltf_display, [])

if not results_for_display:
    st.info("Realiza un análisis o ajusta los filtros si no se encontraron pares. Revisa los logs si el problema persiste.")
else:
    # Preparar datos para la tabla (similar a la v3, adaptado a las claves de CervelloStrategy)
    table_data_list = []
    for result_item in results_for_display:
        if not (result_item and isinstance(result_item, dict)): continue
        
        indicators_item = result_item.get('indicators', {})
        close_price_item = indicators_item.get('Close_LTF', np.nan)
        oi_item = result_item.get('open_interest_value', np.nan)
        fr_item = result_item.get('funding_rate', np.nan)

        table_data_list.append({
            'Par': result_item.get('pair', 'N/A'), 
            'Señal IA': result_item.get('signal', 'N/A'), # Clave 'signal' de Cervello
            'Conf.': f"{result_item.get('confidence', 0.0):.0%}",
            'Precio': f"{close_price_item:.4f}" if not pd.isna(close_price_item) else "N/A",
            'Tend.LTF': result_item.get('trend_ltf', 'N/A'),
            'RSI': f"{indicators_item.get('RSI_LTF', np.nan):.1f}",
            'ADX': f"{indicators_item.get('ADX_LTF', np.nan):.1f}",
            'OI': f"${oi_item:,.0f}" if not pd.isna(oi_item) else "N/A",
            'FR': f"{fr_item*100:.4f}%" if not pd.isna(fr_item) else "N/A",
            'Error': result_item.get('error') # Mostrar errores si los hubo
        })
    
    df_results_table = pd.DataFrame(table_data_list)
    df_ok_results = df_results_table[df_results_table['Error'].isna()].copy()
    df_error_results = df_results_table[df_results_table['Error'].notna()].copy()

    # Función para formatear señal con HTML (la misma que antes)
    def format_signal_cell_html(signal_str):
        # ... (tu función format_signal_html completa aquí)
        s_class = "signal-neutral" 
        if signal_str == 'LONG': s_class = "signal-long"
        elif signal_str == 'SHORT': s_class = "signal-short"
        elif signal_str == 'CAUTIOUS_LONG': s_class = "signal-cautious_long" # Asegúrate que CSS tenga esta clase
        elif signal_str == 'CAUTIOUS_SHORT': s_class = "signal-cautious_short" # Y esta
        return f'<span class="{s_class}">{signal_str}</span>'


    if not df_ok_results.empty:
        st.markdown("##### Pares Analizados por IA Cervello (Sin Errores)")
        df_ok_html_display = df_ok_results.drop(columns=['Error']).copy()
        df_ok_html_display['Señal IA'] = df_ok_html_display['Señal IA'].apply(format_signal_cell_html)
        st.markdown(df_ok_html_display.to_html(escape=False, index=False, justify='center', classes='dataframe-custom'), unsafe_allow_html=True)
        
        # Actualizar lista de pares válidos para el selector de detalles
        available_pairs_for_details = df_ok_results['Par'].tolist()
        if not st.session_state.selected_details_pair and available_pairs_for_details:
            st.session_state.selected_details_pair = available_pairs_for_details[0]
        elif st.session_state.selected_details_pair not in available_pairs_for_details and available_pairs_for_details:
            st.session_state.selected_details_pair = available_pairs_for_details[0]
        elif not available_pairs_for_details:
             st.session_state.selected_details_pair = None # No hay pares para detallar
    else:
        st.info("No hay resultados de análisis sin errores para esta temporalidad.")
        available_pairs_for_details = []
        st.session_state.selected_details_pair = None


    if not df_error_results.empty:
        with st.expander(f"⚠️ Ver {len(df_error_results)} Pares con Errores de Análisis"):
            st.dataframe(df_error_results[['Par', 'Error']], use_container_width=True, hide_index=True)
    st.markdown("---")

    # --- Sección de Detalles del Par ---
    # (Esta sección es muy similar a la v3, asegúrate que las claves que usas para obtener
    #  datos de `selected_data_item` coincidan con lo que devuelve `CervelloStrategy`.
    #  Por ejemplo, `selected_data_item.get('volatility_status_pair')` si Cervello lo añade,
    #  o `selected_data_item.get('market_profile_ প্রভাব')` para ver el perfil aplicado.)

    details_col_selector, details_col_display = st.columns([1, 2.8]) 

    with details_col_selector:
        st.subheader("🔍 Detalles del Par")
        if available_pairs_for_details:
            details_radio_key = f"details_radio_cervello_{active_ltf_display}"
            current_details_idx = 0
            if st.session_state.selected_details_pair in available_pairs_for_details:
                current_details_idx = available_pairs_for_details.index(st.session_state.selected_details_pair)
            
            chosen_details_pair = st.radio(
                "Selecciona un par para ver detalles:", 
                available_pairs_for_details, 
                index=current_details_idx, 
                key=details_radio_key,
                label_visibility="collapsed"
            )
            if chosen_details_pair and chosen_details_pair != st.session_state.selected_details_pair:
                st.session_state.selected_details_pair = chosen_details_pair
                st.rerun() 
            
            if st.session_state.selected_details_pair:
                if st.button(f"➕ Añadir {st.session_state.selected_details_pair} a Seguimiento", key=f"add_to_watch_{st.session_state.selected_details_pair}"):
                    if st.session_state.selected_details_pair not in st.session_state.pairs_to_operate:
                        st.session_state.pairs_to_operate.append(st.session_state.selected_details_pair)
                        st.success(f"{st.session_state.selected_details_pair} añadido a seguimiento.")
                        time.sleep(0.5); st.rerun()
                    else:
                        st.info(f"{st.session_state.selected_details_pair} ya está en seguimiento.")
        else:
            st.write("No hay pares válidos disponibles para mostrar detalles.")

    with details_col_display:
        st.subheader("📊 Análisis Detallado IA Cervello")
        selected_data_item = None
        if st.session_state.selected_details_pair:
            # Encontrar el item de resultado completo para el par seleccionado
            selected_data_item = next((r_item for r_item in results_for_display if isinstance(r_item, dict) and r_item.get('pair') == st.session_state.selected_details_pair and not r_item.get('error')), None)

        if selected_data_item:
            # Extraer todos los datos necesarios de selected_data_item
            pair_name = selected_data_item['pair']
            signal_ia = selected_data_item['signal']
            confidence_ia = selected_data_item.get('confidence', 0.0)
            # ... (extraer trend_ltf, trend_htf, volatility_status_pair, reasoning_ia, market_profile_applied, etc.)
            # ... (extraer indicators_ia, fib_levels_ia, df_ltf_ia, oi_ia, fr_ia, next_fr_ia)

            reasoning_ia = selected_data_item.get('reasoning', [])
            market_profile_applied = selected_data_item.get('market_profile_ प्रभाव', {}) # Usar el nombre de clave correcto

            st.markdown(f"#### {pair_name} - {format_signal_cell_html(signal_ia)} (Conf: {confidence_ia:.0%})", unsafe_allow_html=True)
            # ... (mostrar más captions con trend_ltf, trend_htf, volatility_status_pair)

            with st.expander("🧠 Lógica de Decisión Cervello (Razonamiento)", expanded=True):
                st.caption(f"Perfil de Mercado Global Aplicado: Régimen={market_profile_applied.get('regime','N/A')}, Sent.={market_profile_applied.get('overall_sentiment','N/A')}")
                if reasoning_ia:
                    for reason_text in reasoning_ia:
                        # ... (tu lógica de formateo de colores para el razonamiento) ...
                        if "DECISIÓN CERVELLO:" in reason_text: st.markdown(f"<p style='font-weight:bold;color:#FF9900;'>🎯 {reason_text}</p>", unsafe_allow_html=True)
                        elif "CONF_" in reason_text or "Rebote" in reason_text or "Rechazo" in reason_text: st.markdown(f"<p style='color:#2ECC71;'>✅ {reason_text}</p>", unsafe_allow_html=True)
                        elif "WARN_" in reason_text or "MODIFICADOR:" in reason_text or "Ajuste:" in reason_text: st.markdown(f"<p style='color:#F39C12;'>⚠️ {reason_text}</p>", unsafe_allow_html=True)
                        else: st.markdown(f"<p style='color:#BDC3C7;'>ℹ️ {reason_text}</p>", unsafe_allow_html=True)
                else: st.caption("No hay razonamiento detallado disponible.")
            
            st.markdown("---")
            # Métricas Clave (como en la v3, usando las claves correctas de selected_data_item)
            # ... (m1, m2, m3, m4 con st.metric)

  # app.py (dentro de with details_col_display:, después de st.markdown("---"))

            # --- MÉTRICAS CLAVE ---
            st.subheader("📊 Métricas Clave del Par (LTF)")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4) # 4 columnas para métricas

            # Métrica 1: Precio Actual
            close_price_metric = selected_data_item.get('indicators', {}).get('Close_LTF', np.nan)
            m_col1.metric("Precio Actual", 
                          f"{close_price_metric:.{selected_data_item.get('pair', 'USDT').split('/')[0].upper() == 'SHIB' and 8 or 4}f}" if not pd.isna(close_price_metric) else "N/A") # Ajustar decimales para SHIB

            # Métrica 2: RSI (LTF)
            rsi_metric_val = selected_data_item.get('indicators', {}).get('RSI_LTF', np.nan)
            m_col2.metric("RSI (LTF)", f"{rsi_metric_val:.1f}" if not pd.isna(rsi_metric_val) else "N/A")

            # Métrica 3: Open Interest
            oi_metric_val = selected_data_item.get('open_interest_value', np.nan)
            m_col3.metric("Open Interest", f"${oi_metric_val:,.0f}" if not pd.isna(oi_metric_val) and oi_metric_val > 0 else "N/A")

            # Métrica 4: Funding Rate
            fr_metric_val = selected_data_item.get('funding_rate', np.nan)
            next_fr_time_metric = selected_data_item.get('next_funding_time', None)
            fr_display_str = "N/A"
            if not pd.isna(fr_metric_val):
                fr_display_str = f"{fr_metric_val*100:.4f}%"
                if next_fr_time_metric and not pd.isna(next_fr_time_metric):
                    try:
                        fr_time_dt = pd.to_datetime(next_fr_time_metric, unit='ms', utc=True)
                        fr_display_str += f" (Prox: {fr_time_dt.strftime('%H:%M %Z')})"
                    except: pass # Ignorar error de formato de fecha
            m_col4.metric("Funding Rate", fr_display_str)
            
            st.markdown("---") # Otro separador

            #app.py (dentro de with details_col_display:)

            # ... (después de las Métricas Clave y antes de los Expanders de Fibo/Indicadores) ...
            st.markdown("---")
            st.subheader("🎯 Objetivos de Precio Sugeridos por IA")
            
            price_targets_data = selected_data_item.get('price_targets', {})
            current_signal_for_targets = selected_data_item.get('signal', 'NEUTRAL')

            if current_signal_for_targets not in ['NEUTRAL', 'ERROR'] and price_targets_data:
                pt_col1, pt_col2, pt_col3, pt_col_sl = st.columns(4)
                
                tp1_val = price_targets_data.get('tp1', np.nan)
                tp2_val = price_targets_data.get('tp2', np.nan)
                tp3_val = price_targets_data.get('tp3', np.nan)
                sl_val = price_targets_data.get('sl', np.nan)

                # Ajustar decimales dinámicamente
                pair_base = selected_data_item.get('pair', 'USDT').split('/')[0].upper()
                price_decimals = 8 if pair_base in ['SHIB', 'PEPE', 'BONK'] else 4 # Ejemplo
                target_decimals = 8 if pair_base in ['SHIB', 'PEPE', 'BONK'] else 2 # Objetivos con menos decimales

                pt_col1.metric("Take Profit 1", 
                               f"{tp1_val:.{target_decimals}f}" if not pd.isna(tp1_val) else "N/A",
                               delta_color="normal") # o "inverse" si quieres verde/rojo
                pt_col2.metric("Take Profit 2", 
                               f"{tp2_val:.{target_decimals}f}" if not pd.isna(tp2_val) else "N/A",
                               delta_color="normal")
                pt_col3.metric("Take Profit 3 / Ext.", 
                               f"{tp3_val:.{target_decimals}f}" if not pd.isna(tp3_val) else "N/A",
                               delta_color="normal")
                pt_col_sl.metric("Stop Loss", 
                                 f"{sl_val:.{target_decimals}f}" if not pd.isna(sl_val) else "N/A",
                                 delta_color="inverse" if current_signal_for_targets.startswith("LONG") else "normal")
                st.caption("Objetivos calculados basados en ATR y/o niveles Fibonacci/Swing. Usar con discreción y gestión de riesgo propia.")
            elif current_signal_for_targets in ['NEUTRAL', 'ERROR']:
                st.caption("No se calculan objetivos para señales NEUTRALES o con ERRORES.")
            else:
                st.caption("Datos de objetivos de precio no disponibles para esta señal.")
            
            st.markdown("---") # Otro separador



            # --- EXPANDER DE NIVELES FIBONACCI ---
            fib_levels_data = selected_data_item.get('fib_levels', {})
            with st.expander("🔑 Niveles Fibonacci Calculados (LTF)", expanded=False):
                if fib_levels_data and isinstance(fib_levels_data, dict) and any(not pd.isna(v) for v in fib_levels_data.values()):
                    # Ordenar niveles por el valor del nivel (0.0, 23.6, etc.)
                    sorted_fib_items = sorted(fib_levels_data.items(), key=lambda item: item[0])
                    
                    # Determinar cuántas columnas usar, máximo 4-5 para legibilidad
                    num_fib_levels_to_show = len([v for v in fib_levels_data.values() if not pd.isna(v)])
                    fib_cols_num = min(num_fib_levels_to_show, 5) 
                    if fib_cols_num > 0:
                        fib_cols = st.columns(fib_cols_num)
                        col_idx = 0
                        for level_pct, price_at_level in sorted_fib_items:
                            if not pd.isna(price_at_level):
                                fib_label = f"Fib {level_pct:.1f}%"
                                fib_value = f"{price_at_level:.{selected_data_item.get('pair', 'USDT').split('/')[0].upper() == 'SHIB' and 8 or 2}f}" # Más decimales para SHIB, 2 para otros
                                fib_cols[col_idx % fib_cols_num].metric(label=fib_label, value=fib_value)
                                col_idx += 1
                        st.caption(f"Basado en el último swing de {getattr(config, 'FIB_LOOKBACK', 100)} velas en LTF.")
                    else:
                        st.caption("No hay niveles Fibonacci válidos para mostrar (todos NaN).")
                else:
                    st.caption("Niveles Fibonacci no disponibles o no calculados para este par.")

            # --- EXPANDER DE VALORES DE INDICADORES DETALLADOS ---
            all_indicators_data = selected_data_item.get('indicators', {})
            with st.expander("⚙️ Valores de Indicadores Detallados (LTF)", expanded=False):
                if all_indicators_data and isinstance(all_indicators_data, dict):
                    # Filtrar NaNs y formatear para una mejor visualización en JSON
                    displayable_indicators = {}
                    for key, value in sorted(all_indicators_data.items()): # Ordenar alfabéticamente
                        if value is not None and not (isinstance(value, float) and math.isnan(value)):
                            if isinstance(value, (float, np.floating)):
                                displayable_indicators[key] = f"{value:.4f}" # Formatear floats
                            else:
                                displayable_indicators[key] = str(value)
                    
                    if displayable_indicators:
                        st.json(displayable_indicators, expanded=False)
                    else:
                        st.caption("No hay valores de indicadores válidos para mostrar.")
                else:
                    st.caption("Datos de indicadores no disponibles.")

            # Gráfico Interactivo (como en la v3, usando el df_ltf_ia de selected_data_item)
			
            # app.py (dentro de with details_col_display:, después de los expanders de Fibo e Indicadores)

            df_for_chart = selected_data_item.get('df_ltf') # DataFrame con indicadores
            pair_name_for_chart = selected_data_item.get('pair', 'N/A')
            current_ltf_for_chart = active_ltf_display # La TF seleccionada

            if isinstance(df_for_chart, pd.DataFrame) and not df_for_chart.empty:
                with st.expander("📈 Gráfico Interactivo Avanzado (LTF)", expanded=True):
                    if not all(col in df_for_chart.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                        st.warning(f"DataFrame para {pair_name_for_chart} no contiene todas las columnas OHLCV requeridas para el gráfico.")
                    else:
                        try:
                            # --- Nombres de Columnas de Indicadores (deben coincidir con IndicatorCalculator) ---
                            rsi_col = f'RSI_{config.RSI_PERIOD}'
                            macd_col = f'MACD_{config.MACD_F}_{config.MACD_S}_{config.MACD_SIG}'
                            macds_col = f'MACDs_{config.MACD_F}_{config.MACD_S}_{config.MACD_SIG}'
                            macdh_col = f'MACDh_{config.MACD_F}_{config.MACD_S}_{config.MACD_SIG}'
                            sma_s_col = f'SMA_{config.SMA_S_P}'
                            sma_l_col = f'SMA_{config.SMA_L_P}'
                            ema_s_col = f'EMA_{config.EMA_S_P}'
                            ema_l_col = f'EMA_{config.EMA_L_P}'
                            bbu_col = f'BBU_{config.BB_P}_{config.BB_STD}'
                            bbm_col = f'BBM_{config.BB_P}_{config.BB_STD}'
                            bbl_col = f'BBL_{config.BB_P}_{config.BB_STD}'
                            vol_ma_col = f'volume_ma_{config.VOL_MA_P}'
                            # Ichimoku (si los usas y calculas consistentemente)
                            isa_col = f'ISA_{config.ICHIMOKU_TENKAN}_{config.ICHIMOKU_KIJUN}_{config.ICHIMOKU_SENKOU_B}'
                            isb_col = f'ISB_{config.ICHIMOKU_TENKAN}_{config.ICHIMOKU_KIJUN}_{config.ICHIMOKU_SENKOU_B}'
                            its_col = f'ITS_{config.ICHIMOKU_TENKAN}_{config.ICHIMOKU_KIJUN}_{config.ICHIMOKU_SENKOU_B}' # Tenkan
                            iks_col = f'IKS_{config.ICHIMOKU_TENKAN}_{config.ICHIMOKU_KIJUN}_{config.ICHIMOKU_SENKOU_B}' # Kijun
                            
                            # Crear figura con subplots
                            fig = make_subplots(
                                rows=5, cols=1, shared_xaxes=True,
                                vertical_spacing=0.015, # Espacio vertical reducido
                                row_heights=[0.50, 0.12, 0.12, 0.12, 0.14] # Ajustar alturas para más subplots
                            )

                            # --- Subplot 1: Precio, Medias Móviles, Bollinger Bands, Ichimoku, Fibonacci ---
                            # Velas
                            fig.add_trace(go.Candlestick(
                                x=df_for_chart.index, open=df_for_chart['open'], high=df_for_chart['high'],
                                low=df_for_chart['low'], close=df_for_chart['close'], name='Precio',
                                increasing_line_color='#2ECC71', decreasing_line_color='#E74C3C'
                            ), row=1, col=1)

							# app.py (dentro de la sección del gráfico, después de fig.add_trace(go.Candlestick(...)))

                            # --- Añadir Líneas de Take Profit y Stop Loss al Gráfico ---
                            price_targets_chart = selected_data_item.get('price_targets', {})
                            signal_for_chart_lines = selected_data_item.get('signal', 'NEUTRAL')

                            if signal_for_chart_lines not in ['NEUTRAL', 'ERROR'] and price_targets_chart:
                                tp_color = "rgba(0, 200, 0, 0.7)"  # Verde para TPs
                                sl_color = "rgba(200, 0, 0, 0.7)"  # Rojo para SL

                                for i, (level_key, level_price) in enumerate(price_targets_chart.items()):
                                    if not pd.isna(level_price) and level_price > 0:
                                        line_name = ""
                                        line_color_current = tp_color
                                        line_dash_style = "dash"

                                        if level_key == 'sl':
                                            line_name = "Stop Loss"
                                            line_color_current = sl_color
                                            line_dash_style = "solid" # SL más prominente
                                        elif level_key.startswith('tp'):
                                            line_name = f"Take Profit {level_key[-1]}"
                                            # Podrías variar el dash para diferentes TPs
                                            if level_key == 'tp1': line_dash_style = "dot"
                                            elif level_key == 'tp2': line_dash_style = "dashdot"
                                        
                                        if line_name:
                                            fig.add_hline(
                                                y=level_price,
                                                line_dash=line_dash_style,
                                                line_color=line_color_current,
                                                annotation_text=f" {line_name}: {level_price:.{target_decimals}f}", # Usar target_decimals
                                                annotation_position="bottom right" if level_key.startswith('tp') else "top right",
                                                annotation_font_size=10,
                                                annotation_font_color=line_color_current,
                                                row=1, col=1 # Añadir al subplot de precios
                                            )
                            # Medias Móviles (EMA y SMA)
                            if ema_s_col in df_for_chart.columns: fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[ema_s_col], name=f'EMA {config.EMA_S_P}', line=dict(color='cyan', width=1)), row=1, col=1)
                            if ema_l_col in df_for_chart.columns: fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[ema_l_col], name=f'EMA {config.EMA_L_P}', line=dict(color='magenta', width=1.2)), row=1, col=1)
                            # if sma_s_col in df_for_chart.columns: fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[sma_s_col], name=f'SMA {config.SMA_S_P}', line=dict(color='yellow', width=1, dash='dot')), row=1, col=1)
                            # if sma_l_col in df_for_chart.columns: fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[sma_l_col], name=f'SMA {config.SMA_L_P}', line=dict(color='orange', width=1.2, dash='dot')), row=1, col=1)

                            # Bandas de Bollinger
                            if bbu_col in df_for_chart.columns: fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[bbu_col], name='BB Sup', line=dict(color='rgba(150,150,150,0.4)', width=1)), row=1, col=1)
                            if bbl_col in df_for_chart.columns: fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[bbl_col], name='BB Inf', line=dict(color='rgba(150,150,150,0.4)', width=1), fill='tonexty', fillcolor='rgba(150,150,150,0.05)'), row=1, col=1) # Relleno entre BBU y BBL
                            if bbm_col in df_for_chart.columns: fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[bbm_col], name='BB Med', line=dict(color='rgba(255,165,0,0.5)', width=1, dash='dashdot')), row=1, col=1)
                            
                            # Ichimoku Cloud (Kumo) y líneas
                            if all(c in df_for_chart.columns for c in [isa_col, isb_col, its_col, iks_col]):
                                # Kumo (Nube)
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[isa_col], name='Senkou A', line=dict(color='rgba(0,255,0,0.2)', width=1), fill=None), row=1, col=1)
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[isb_col], name='Senkou B', line=dict(color='rgba(255,0,0,0.2)', width=1), fill='tonexty', 
                                                         fillcolor='rgba(128,128,128,0.1)'), row=1, col=1) # Relleno entre Senkou A y B
                                # Tenkan-sen y Kijun-sen
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[its_col], name='Tenkan', line=dict(color='blue', width=1)), row=1, col=1)
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[iks_col], name='Kijun', line=dict(color='red', width=1)), row=1, col=1)
                                # Chikou Span (desplazado hacia atrás)
                                # ics_col = f'ICS_{config.ICHIMOKU_TENKAN}_{config.ICHIMOKU_KIJUN}_{config.ICHIMOKU_SENKOU_B}'
                                # if ics_col in df_for_chart.columns:
                                #     chikou_displaced = df_for_chart[ics_col].shift(-config.ICHIMOKU_KIJUN) # Desplazar Kijun periodos hacia atrás
                                #     fig.add_trace(go.Scatter(x=df_for_chart.index, y=chikou_displaced, name='Chikou', line=dict(color='green', width=1, dash='dot')), row=1, col=1)


                            # Niveles Fibonacci (del análisis de estrategia)
                            fib_levels_from_strategy = selected_data_item.get('fib_levels', {}) # Usar los fibs del resultado del análisis
                            if fib_levels_from_strategy:
                                for level_pct, price_val in sorted(fib_levels_from_strategy.items()):
                                    if price_val is not None and not pd.isna(price_val):
                                        fig.add_hline(y=price_val, line_dash="dash", 
                                                      line_color="rgba(255,153,0,0.6)", 
                                                      annotation_text=f" Fib {level_pct:.1f}%", 
                                                      annotation_position="bottom right", row=1, col=1)
                            
                            # --- Subplot 2: RSI ---
                            if rsi_col in df_for_chart.columns:
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[rsi_col], name='RSI', line_color='#00BCD4'), row=2, col=1)
                                fig.add_hline(y=config.RSI_OB, line_dash="dot", line_color="rgba(233,30,99,0.7)", name=f'OB {config.RSI_OB}', row=2, col=1)
                                fig.add_hline(y=config.RSI_OS, line_dash="dot", line_color="rgba(76,175,80,0.7)", name=f'OS {config.RSI_OS}', row=2, col=1)
                            else: logger.warning(f"Columna RSI '{rsi_col}' no encontrada para el gráfico.")

                            # --- Subplot 3: MACD ---
                            if all(c in df_for_chart.columns for c in [macd_col, macds_col, macdh_col]):
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[macd_col], name='MACD', line_color='#FFC107'), row=3, col=1) # Amarillo para MACD
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[macds_col], name='MACD Sig', line_color='#F44336'), row=3, col=1) # Rojo para Signal
                                hist_colors = ['#4CAF50' if v >= 0 else '#E91E63' for v in df_for_chart[macdh_col].fillna(0)] # Rellenar NaNs para colores
                                fig.add_trace(go.Bar(x=df_for_chart.index, y=df_for_chart[macdh_col], name='MACD Hist', marker_color=hist_colors), row=3, col=1)
                            else: logger.warning(f"Columnas MACD ({macd_col}, etc.) no encontradas para el gráfico.")

                            # --- Subplot 4: ADX (Fuerza de Tendencia) ---
                            adx_val_col = f'ADX_{config.ADX_P}'
                            adx_plus_di_col = f'DMP_{config.ADX_P}' # +DI
                            adx_minus_di_col = f'DMN_{config.ADX_P}'# -DI
                            if all(c in df_for_chart.columns for c in [adx_val_col, adx_plus_di_col, adx_minus_di_col]):
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[adx_val_col], name='ADX', line_color='rgba(200,200,200,0.8)'), row=4, col=1)
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[adx_plus_di_col], name='+DI', line_color='rgba(0,200,0,0.6)'), row=4, col=1)
                                fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[adx_minus_di_col], name='-DI', line_color='rgba(200,0,0,0.6)'), row=4, col=1)
                                fig.add_hline(y=config.CERVELLO_ADX_THRESHOLD, line_dash="dashdot", line_color="rgba(255,255,255,0.3)", name=f'ADX Umbral ({config.CERVELLO_ADX_THRESHOLD})', row=4, col=1)
                            else: logger.warning(f"Columnas ADX ({adx_val_col}, etc.) no encontradas para el gráfico.")
                                
                            # --- Subplot 5: Volumen y MA de Volumen ---
                            if 'volume' in df_for_chart.columns:
                                vol_bar_colors = ['rgba(0,180,0,0.6)' if c >= o else 'rgba(180,0,0,0.6)' for c, o in zip(df_for_chart['close'].fillna(0), df_for_chart['open'].fillna(0))]
                                fig.add_trace(go.Bar(x=df_for_chart.index, y=df_for_chart['volume'], name='Volumen', marker_color=vol_bar_colors), row=5, col=1)
                                if vol_ma_col in df_for_chart.columns:
                                    fig.add_trace(go.Scatter(x=df_for_chart.index, y=df_for_chart[vol_ma_col], name=f'Vol MA({config.VOL_MA_P})', line_color='rgba(255,152,0,0.8)'), row=5, col=1)
                            else: logger.warning("Columna 'volume' no encontrada para el gráfico.")


                            # --- Anotaciones de Señal (Ejemplo Básico) ---
                            # Esto podría ser más sofisticado, marcando la vela donde se generó la señal si tuvieras esa info.
                            # Por ahora, una anotación general basada en la señal actual.
                            current_signal_ia = selected_data_item.get('signal', 'N/A')
                            if current_signal_ia not in ['NEUTRAL', 'N/A', 'ERROR'] and len(df_for_chart) > 1:
                                last_candle_time = df_for_chart.index[-1]
                                last_close_price = df_for_chart['close'].iloc[-1]
                                annotation_y_offset = (df_for_chart['high'].iloc[-1] - df_for_chart['low'].iloc[-1]) * 0.5 # Pequeño offset
                                
                                signal_arrow = "▲" if "LONG" in current_signal_ia else "▼" if "SHORT" in current_signal_ia else ""
                                signal_color = "green" if "LONG" in current_signal_ia else "red" if "SHORT" in current_signal_ia else "grey"
                                
                                if signal_arrow:
                                    fig.add_annotation(
                                        x=last_candle_time,
                                        y=last_close_price + (annotation_y_offset if "SHORT" in current_signal_ia else -annotation_y_offset),
                                        text=f"{signal_arrow} {current_signal_ia}",
                                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                                        ax=0, ay=-40 if "LONG" in current_signal_ia else 40, # Offset de la flecha
                                        font=dict(size=12, color=signal_color),
                                        bordercolor="#c7c7c7", borderwidth=1, borderpad=4,
                                        bgcolor="rgba(20,20,20,0.7)",
                                        opacity=0.8,
                                        row=1, col=1
                                    )

                            # --- Layout Final del Gráfico ---
                            fig.update_layout(
                                title_text=f"Análisis Avanzado IA: {pair_name_for_chart} - {current_ltf_for_chart} | Señal: {current_signal_ia}",
                                height=950, # Aumentar altura para más subplots
                                template="plotly_dark",
                                xaxis_rangeslider_visible=False,
                                hovermode="x unified", # Información unificada al pasar el ratón
                                legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="right", x=1, traceorder="normal"),
                                margin=dict(l=40, r=40, t=120, b=30) # Ajustar márgenes
                            )
                            # Actualizar títulos de ejes Y
                            fig.update_yaxes(title_text="Precio / Indicadores", row=1, col=1, title_standoff=5)
                            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, title_standoff=5)
                            fig.update_yaxes(title_text="MACD", row=3, col=1, title_standoff=5)
                            fig.update_yaxes(title_text="ADX/DI", row=4, col=1, title_standoff=5)
                            fig.update_yaxes(title_text="Volumen", row=5, col=1, title_standoff=5)
                            
                            # Ocultar ejes X de subplots superiores para un look más limpio
                            fig.update_xaxes(showticklabels=False, row=1, col=1)
                            fig.update_xaxes(showticklabels=False, row=2, col=1)
                            fig.update_xaxes(showticklabels=False, row=3, col=1)
                            fig.update_xaxes(showticklabels=False, row=4, col=1)
                            fig.update_xaxes(showticklabels=True, type='category', row=5, col=1) # Mostrar ticks en el último

                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e_graph:
                            st.error(f"Error al generar el gráfico para {pair_name_for_chart}: {e_graph}")
                            logger.error(f"Error generando gráfico para {pair_name_for_chart}: {e_graph}", exc_info=True)
            
            else: # Si df_for_chart no es válido
                st.warning(f"No hay datos de DataFrame válidos (df_ltf) disponibles para mostrar el gráfico de {selected_data_item.get('pair','este par')}.")
        
     
