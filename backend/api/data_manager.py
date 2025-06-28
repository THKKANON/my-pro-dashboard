import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
import warnings
import time
import threading
import requests
import json
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 설정 ---
CHART_TYPE = 'Heikin-Ashi'
FETCH_COUNT = 200
COIN_LIST = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'TRXUSDT', 'ETCUSDT', 'BCHUSDT']
TIMEFRAME_OPTIONS = ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d']
STOCH_OVERBOUGHT, STOCH_OVERSOLD = 80, 20
KERNEL_RSI_LENGTH, KERNEL_RSI_BANDWIDTH = 14, 4
K_RSI_LONG_ENTRY, K_RSI_SHORT_ENTRY = 30, 70
KLINE_REFRESH_INTERVAL_SECONDS = 3600

TELEGRAM_BOT_TOKEN = "여기에_텔레그램_봇_토큰을_붙여넣으세요"
TELEGRAM_CHAT_ID = "여기에_텔레그램_채팅_ID를_붙여넣으세요"

client = Client()

# --- 전역 데이터 캐시 및 잠금 ---
app_data_cache = {}
lock = threading.Lock()

# --- 함수들 ---
def send_telegram_notification(message):
    token, chat_id = TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if token.startswith("여기에") or chat_id.startswith("여기에"): return
    try: requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={'chat_id': chat_id, 'text': message}).raise_for_status()
    except requests.exceptions.RequestException as e: print(f"Telegram Error: {e}")

def calculate_heikin_ashi(df):
    ha_df = df.copy(); ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    for i in range(len(ha_df)):
        if i == 0: ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (df.iloc[i]['Open'] + df.iloc[i]['Close']) / 2
        else: ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (ha_df.iloc[i-1]['Open'] + ha_df.iloc[i-1]['Close']) / 2
    ha_df['High'] = ha_df[['High', 'Open', 'Close']].max(axis=1); ha_df['Low'] = ha_df[['Low', 'Open', 'Close']].min(axis=1)
    return ha_df

def get_binance_futures_candles(symbol, interval, limit):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time','Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume', 'Ignore'])
        df = df[['Open_time', 'Open', 'High', 'Low', 'Close']].astype(float); df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
        df.set_index('Date', inplace=True)
        df.index.name = 'Date'
        df = df.tz_localize('UTC').tz_convert('Asia/Seoul')
        return df
    except (BinanceAPIException, Exception) as e:
        print(f"Error fetching klines for {symbol}-{interval}: {e}"); return None

def kernel_regression(data, bandwidth):
    n = len(data); y_hat = np.zeros(n)
    for i in range(n):
        kernels = np.exp(-((np.arange(n) - i) / bandwidth)**2 / 2); weights = kernels / np.sum(kernels)
        y_hat[i] = np.sum(weights * data)
    return y_hat

def find_stoch_rsi_signals(stoch_rsi):
    k, d = stoch_rsi['STOCHRSIk_14_14_3_3'], stoch_rsi['STOCHRSId_14_14_3_3']; prev_k, prev_d = k.shift(1), d.shift(1)
    buy_signals = (k > d) & (prev_k <= prev_d) & (d < STOCH_OVERSOLD); sell_signals = (k < d) & (prev_k >= prev_d) & (d > STOCH_OVERBOUGHT)
    return buy_signals, sell_signals

def find_kernel_rsi_signals(kernel_rsi):
    prev_k_rsi = kernel_rsi.shift(1)
    long_permission = (kernel_rsi > K_RSI_LONG_ENTRY) & (prev_k_rsi <= K_RSI_LONG_ENTRY)
    short_permission = (kernel_rsi < K_RSI_SHORT_ENTRY) & (prev_k_rsi >= K_RSI_SHORT_ENTRY)
    return long_permission, short_permission

# ✨✨✨ 핵심 수정: 누락되었던 get_interval_seconds 함수 추가 ✨✨✨
def get_interval_seconds(interval_str):
    """'1m', '1h', '1d' 같은 문자열을 초 단위로 변환"""
    if not interval_str: return 0
    unit = interval_str[-1]
    try:
        value = int(interval_str[:-1])
        if unit == 'm': return value * 60
        if unit == 'h': return value * 3600
        if unit == 'd': return value * 86400
    except (ValueError, IndexError):
        return 0
    return 0
    
def process_dataframe(df):
    stoch_rsi = df.ta.stochrsi()
    stoch_buy, stoch_sell = find_stoch_rsi_signals(stoch_rsi)
    rsi = df.ta.rsi(length=KERNEL_RSI_LENGTH)
    if rsi.dropna().empty: return None
    kernel_rsi = pd.Series(kernel_regression(rsi.dropna().values, KERNEL_RSI_BANDWIDTH), index=rsi.dropna().index)
    krsi_long, krsi_short = find_kernel_rsi_signals(kernel_rsi)
    plot_df = calculate_heikin_ashi(df) if CHART_TYPE == 'Heikin-Ashi' else df
    return {'plot_df': plot_df, 'stoch_rsi': stoch_rsi, 'stoch_buy': stoch_buy, 'stoch_sell': stoch_sell, 'kernel_rsi': kernel_rsi, 'krsi_long': krsi_long, 'krsi_short': krsi_short}

# --- 하이브리드 방식 워커 스레드 ---
def data_worker():
    kline_data_cache = {}
    last_kline_fetch_time = 0
    notification_tracker = {}

    while True:
        now = time.time()
        
        if now - last_kline_fetch_time > KLINE_REFRESH_INTERVAL_SECONDS:
            print(f"[{time.strftime('%H:%M:%S')}] Worker: Performing full kline data refresh...")
            for symbol in COIN_LIST:
                if symbol not in kline_data_cache: kline_data_cache[symbol] = {}
                for interval in TIMEFRAME_OPTIONS:
                    kline_data_cache[symbol][interval] = get_binance_futures_candles(symbol, interval, FETCH_COUNT)
                    time.sleep(0.05)
            last_kline_fetch_time = now
            print(f"[{time.strftime('%H:%M:%S')}] Worker: Full kline data refreshed.")

        try:
            tickers = client.futures_ticker()
            latest_prices = {item['symbol']: float(item['lastPrice']) for item in tickers}
        except Exception as e:
            print(f"Error fetching tickers: {e}"); time.sleep(1); continue

        for symbol in COIN_LIST:
            latest_price = latest_prices.get(symbol)
            if not latest_price or symbol not in kline_data_cache: continue

            processed_data_package = {}
            for interval in TIMEFRAME_OPTIONS:
                df = kline_data_cache[symbol].get(interval)
                if df is None or df.empty: continue
                
                df_copy = df.copy()
                last_candle_time = df_copy.index[-1]
                interval_seconds = get_interval_seconds(interval)
                if interval_seconds == 0: continue # 유효하지 않은 인터벌이면 건너뜀

                now_kst = datetime.fromtimestamp(now, tz=timezone(timedelta(hours=9)))
                
                is_new_candle = False
                if now_kst >= last_candle_time + timedelta(seconds=interval_seconds):
                    is_new_candle = True
                    new_candle_open_time = last_candle_time + timedelta(seconds=interval_seconds)
                    new_row_data = {'Open': df_copy.iloc[-1]['Close'], 'High': latest_price, 'Low': latest_price, 'Close': latest_price}
                    new_row = pd.DataFrame([new_row_data], index=[new_candle_open_time])
                    new_row.index.name = 'Date'
                    df_copy = pd.concat([df_copy, new_row])
                    if len(df_copy) > FETCH_COUNT:
                        df_copy = df_copy.iloc[1:]
                    kline_data_cache[symbol][interval] = df_copy
                else:
                    df_copy.iloc[-1, df_copy.columns.get_loc('Close')] = latest_price
                    df_copy.iloc[-1, df_copy.columns.get_loc('High')] = max(df_copy.iloc[-1]['High'], latest_price)
                    df_copy.iloc[-1, df_copy.columns.get_loc('Low')] = min(df_copy.iloc[-1]['Low'], latest_price)

                processed_data = process_dataframe(df_copy)
                
                if processed_data and is_new_candle:
                    with lock:
                        if not processed_data['krsi_long'].empty and processed_data['krsi_long'].iloc[-2]:
                             signal_key = f"{symbol}_{interval}_long"
                             if time.time() - notification_tracker.get(signal_key, 0) > 300:
                                 send_telegram_notification(f"🚀 [매수 신호] {symbol} ({interval})")
                                 notification_tracker[signal_key] = time.time()
                        if not processed_data['krsi_short'].empty and processed_data['krsi_short'].iloc[-2]:
                             signal_key = f"{symbol}_{interval}_short"
                             if time.time() - notification_tracker.get(signal_key, 0) > 300:
                                send_telegram_notification(f"🔻 [매도 신호] {symbol} ({interval})")
                                notification_tracker[signal_key] = time.time()

                processed_data_package[interval] = processed_data

            with lock:
                app_data_cache[symbol] = processed_data_package
        
        time.sleep(1)

def start_data_workers():
    worker = threading.Thread(target=data_worker, daemon=True)
    worker.start()