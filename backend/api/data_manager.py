# data_manager.py
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import numpy as np
import warnings
import time
import threading
import requests

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- User Settings ---
CHART_TYPE = 'Heikin-Ashi'
FETCH_COUNT = 150
COIN_LIST = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'TRXUSDT', 'ETCUSDT', 'BCHUSDT']
TIMEFRAME_OPTIONS = ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d']
STOCH_OVERBOUGHT, STOCH_OVERSOLD = 80, 20
KERNEL_RSI_LENGTH, KERNEL_RSI_BANDWIDTH = 14, 4
K_RSI_LONG_ENTRY, K_RSI_SHORT_ENTRY = 30, 70

TELEGRAM_BOT_TOKEN = "여기에_텔레그램_봇_토큰을_붙여넣으세요"
TELEGRAM_CHAT_ID = "여기에_텔레그램_채팅_ID를_붙여넣으세요"

client = Client()

# --- 전역 데이터 캐시 및 잠금 ---
app_data_cache = {}
lock = threading.Lock()

# --- Functions ---
def send_telegram_notification(message):
    token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    if token.startswith("여기에") or chat_id.startswith("여기에"):
        print(f"Telegram Token/ID not set. Notification for '{message}' skipped.")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {'chat_id': chat_id, 'text': message}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        print(f"Telegram Notification Sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Telegram message: {e}")

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
        df.set_index('Date', inplace=True); df = df.tz_localize('UTC').tz_convert('Asia/Seoul')
        return df
    except (BinanceAPIException, Exception) as e:
        print(f"Error fetching data for {symbol}-{interval}: {e}"); return None

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

# --- Background Worker Thread ---
def data_worker(symbol, notification_tracker):
    """지정된 한 코인에 대해서, 모든 시간대의 데이터를 계산하는 워커"""
    while True:
        print(f"[{time.strftime('%H:%M:%S')}] Worker ({symbol}): Fetching all timeframes...")
        all_timeframe_data = {}
        for interval in TIMEFRAME_OPTIONS:
            original_df = get_binance_futures_candles(symbol, interval, FETCH_COUNT)
            if original_df is None or original_df.empty:
                all_timeframe_data[interval] = None; continue

            stoch_rsi = original_df.ta.stochrsi(); stoch_buy, stoch_sell = find_stoch_rsi_signals(stoch_rsi)
            rsi = original_df.ta.rsi(length=KERNEL_RSI_LENGTH)
            kernel_rsi = pd.Series(kernel_regression(rsi.dropna().values, KERNEL_RSI_BANDWIDTH), index=rsi.dropna().index)
            krsi_long, krsi_short = find_kernel_rsi_signals(kernel_rsi)
            
            with lock:
                if not krsi_long.empty and krsi_long.iloc[-1]:
                    signal_key = f"{symbol}_{interval}_long";
                    if time.time() - notification_tracker.get(signal_key, 0) > 300:
                        send_telegram_notification(f"🚀 [매수 신호] {symbol} ({interval})"); notification_tracker[signal_key] = time.time()
                if not krsi_short.empty and krsi_short.iloc[-1]:
                    signal_key = f"{symbol}_{interval}_short"
                    if time.time() - notification_tracker.get(signal_key, 0) > 300:
                         send_telegram_notification(f"🔻 [매도 신호] {symbol} ({interval})"); notification_tracker[signal_key] = time.time()
            
            plot_df = calculate_heikin_ashi(original_df) if CHART_TYPE == 'Heikin-Ashi' else original_df
            all_timeframe_data[interval] = {'plot_df': plot_df, 'stoch_rsi': stoch_rsi, 'stoch_buy': stoch_buy, 'stoch_sell': stoch_sell, 'kernel_rsi': kernel_rsi, 'krsi_long': krsi_long, 'krsi_short': krsi_short}
            time.sleep(0.1)
        
        with lock:
            app_data_cache[symbol] = all_timeframe_data
        
        # 실제 업데이트 주기는 이 sleep 시간에 의해 결정됨
        time.sleep(1)

def start_data_workers():
    """모든 코인에 대한 워커 스레드를 생성하고 시작하는 함수"""
    notification_tracker = {}
    for symbol in COIN_LIST:
        worker = threading.Thread(target=data_worker, args=(symbol, notification_tracker), daemon=True)
        worker.start()