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

# --- 설정 (기존과 동일) ---
CHART_TYPE = 'Heikin-Ashi'
FETCH_COUNT = 200
COIN_LIST = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'TRXUSDT', 'ETCUSDT', 'BCHUSDT']
TIMEFRAME_OPTIONS = ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d']
STOCH_OVERBOUGHT, STOCH_OVERSOLD = 80, 20
KERNEL_RSI_LENGTH, KERNEL_RSI_BANDWIDTH = 14, 4
K_RSI_LONG_ENTRY, K_RSI_SHORT_ENTRY = 30, 70
KLINE_REFRESH_INTERVAL_SECONDS = 3600

# 텔레그램 설정은 여기에 직접 입력해주세요.
TELEGRAM_BOT_TOKEN = "여기에_텔레그램_봇_토큰을_붙여넣으세요"
TELEGRAM_CHAT_ID = "여기에_텔레그램_채팅_ID를_붙여넣으세요"

client = Client()

# --- 전역 데이터 캐시 및 잠금 (여러 스레드가 공유) ---
app_data_cache = {}
lock = threading.Lock()

# --- 헬퍼 함수 (기존과 동일) ---
def send_telegram_notification(message):
    token, chat_id = TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    if token.startswith("여기에") or chat_id.startswith("여기에"): return
    try:
        requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={'chat_id': chat_id, 'text': message}).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Telegram Error: {e}")

def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    for i in range(len(ha_df)):
        if i == 0:
            ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (df.iloc[i]['Open'] + df.iloc[i]['Close']) / 2
        else:
            ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (ha_df.iloc[i-1]['Open'] + ha_df.iloc[i-1]['Close']) / 2
    ha_df['High'] = ha_df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['Low'] = ha_df[['Low', 'Open', 'Close']].min(axis=1)
    return ha_df

def get_binance_futures_candles(symbol, interval, limit):
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time','Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume', 'Ignore'])
        df = df[['Open_time', 'Open', 'High', 'Low', 'Close']].astype(float)
        df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
        df.set_index('Date', inplace=True)
        df.index.name = 'Date'
        df = df.tz_localize('UTC').tz_convert('Asia/Seoul')
        return df
    except (BinanceAPIException, Exception) as e:
        print(f"Error fetching klines for {symbol}-{interval}: {e}")
        return None

def kernel_regression(data, bandwidth):
    n = len(data)
    y_hat = np.zeros(n)
    for i in range(n):
        kernels = np.exp(-((np.arange(n) - i) / bandwidth)**2 / 2)
        weights = kernels / np.sum(kernels)
        y_hat[i] = np.sum(weights * data)
    return y_hat

def find_stoch_rsi_signals(stoch_rsi):
    k, d = stoch_rsi['STOCHRSIk_14_14_3_3'], stoch_rsi['STOCHRSId_14_14_3_3']
    prev_k, prev_d = k.shift(1), d.shift(1)
    buy_signals = (k > d) & (prev_k <= prev_d) & (d < STOCH_OVERSOLD)
    sell_signals = (k < d) & (prev_k >= prev_d) & (d > STOCH_OVERBOUGHT)
    return buy_signals, sell_signals

def find_kernel_rsi_signals(kernel_rsi):
    prev_k_rsi = kernel_rsi.shift(1)
    long_permission = (kernel_rsi > K_RSI_LONG_ENTRY) & (prev_k_rsi <= K_RSI_LONG_ENTRY)
    short_permission = (kernel_rsi < K_RSI_SHORT_ENTRY) & (prev_k_rsi >= K_RSI_SHORT_ENTRY)
    return long_permission, short_permission

def get_interval_seconds(interval_str):
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
    if stoch_rsi is None or stoch_rsi.empty: return None # stoch_rsi 계산 불가시 None 반환

    stoch_buy, stoch_sell = find_stoch_rsi_signals(stoch_rsi)
    
    rsi = df.ta.rsi(length=KERNEL_RSI_LENGTH)
    if rsi is None or rsi.dropna().empty: return None # rsi 계산 불가시 None 반환
    
    kernel_rsi_values = kernel_regression(rsi.dropna().values, KERNEL_RSI_BANDWIDTH)
    kernel_rsi = pd.Series(kernel_rsi_values, index=rsi.dropna().index)
    
    krsi_long, krsi_short = find_kernel_rsi_signals(kernel_rsi)
    
    plot_df = calculate_heikin_ashi(df) if CHART_TYPE == 'Heikin-Ashi' else df
    
    return {
        'plot_df': plot_df,
        'stoch_rsi': stoch_rsi,
        'stoch_buy': stoch_buy,
        'stoch_sell': stoch_sell,
        'kernel_rsi': kernel_rsi,
        'krsi_long': krsi_long,
        'krsi_short': krsi_short
    }


# --- ✨✨✨ 핵심 수정: 코인별 데이터 처리를 위한 클래스 정의 ✨✨✨ ---
class CoinWorker(threading.Thread):
    def __init__(self, symbol, client):
        super().__init__()
        self.symbol = symbol
        self.client = client
        self.daemon = True  # 메인 스레드 종료시 함께 종료
        self.kline_data_cache = {}
        self.last_kline_fetch_time = 0
        self.notification_tracker = {}

    def run(self):
        """스레드가 시작되면 실행될 메인 로직"""
        print(f"[{time.strftime('%H:%M:%S')}] Worker started for {self.symbol}")

        while True:
            now = time.time()
            
            # 1. 주기적인 전체 데이터 갱신 (기존 로직과 동일)
            if now - self.last_kline_fetch_time > KLINE_REFRESH_INTERVAL_SECONDS:
                print(f"[{time.strftime('%H:%M:%S')}] {self.symbol}: Performing full kline data refresh...")
                for interval in TIMEFRAME_OPTIONS:
                    self.kline_data_cache[interval] = get_binance_futures_candles(self.symbol, interval, FETCH_COUNT)
                    time.sleep(0.05) # API 호출 간격
                self.last_kline_fetch_time = now
                print(f"[{time.strftime('%H:%M:%S')}] {self.symbol}: Full kline data refreshed.")

            # 2. 실시간 가격 반영 및 신호 생성 (기존 로직과 동일)
            try:
                # 단일 티커 정보만 가져와서 효율성 증대
                ticker = self.client.futures_ticker(symbol=self.symbol)
                latest_price = float(ticker['lastPrice'])
            except Exception as e:
                print(f"Error fetching ticker for {self.symbol}: {e}")
                time.sleep(1)
                continue

            processed_data_package = {}
            for interval in TIMEFRAME_OPTIONS:
                df = self.kline_data_cache.get(interval)
                if df is None or df.empty:
                    continue
                
                df_copy = df.copy()
                last_candle_time = df_copy.index[-1]
                interval_seconds = get_interval_seconds(interval)
                if interval_seconds == 0: continue

                now_kst = datetime.fromtimestamp(now, tz=timezone(timedelta(hours=9)))
                
                is_new_candle = False
                if now_kst >= last_candle_time + timedelta(seconds=interval_seconds):
                    is_new_candle = True
                    # 새 캔들 데이터 추가 및 가장 오래된 데이터 제거
                    new_candle_open_time = last_candle_time + timedelta(seconds=interval_seconds)
                    new_row_data = {'Open': df_copy.iloc[-1]['Close'], 'High': latest_price, 'Low': latest_price, 'Close': latest_price}
                    new_row = pd.DataFrame([new_row_data], index=[new_candle_open_time])
                    new_row.index.name = 'Date'
                    df_copy = pd.concat([df_copy, new_row]).iloc[1:]
                    self.kline_data_cache[interval] = df_copy
                else:
                    # 마지막 캔들 실시간 업데이트
                    df_copy.iloc[-1, df_copy.columns.get_loc('Close')] = latest_price
                    df_copy.iloc[-1, df_copy.columns.get_loc('High')] = max(df_copy.iloc[-1]['High'], latest_price)
                    df_copy.iloc[-1, df_copy.columns.get_loc('Low')] = min(df_copy.iloc[-1]['Low'], latest_price)

                processed_data = process_dataframe(df_copy)
                
                # 새 캔들 생성 시에만 텔레그램 알림 발송 (기존 로직과 동일)
                if processed_data and is_new_candle:
                    if not processed_data['krsi_long'].empty and processed_data['krsi_long'].iloc[-2]:
                         signal_key = f"{self.symbol}_{interval}_long"
                         if time.time() - self.notification_tracker.get(signal_key, 0) > 300:
                             send_telegram_notification(f"🚀 [매수 신호] {self.symbol} ({interval})")
                             self.notification_tracker[signal_key] = time.time()
                    if not processed_data['krsi_short'].empty and processed_data['krsi_short'].iloc[-2]:
                         signal_key = f"{self.symbol}_{interval}_short"
                         if time.time() - self.notification_tracker.get(signal_key, 0) > 300:
                            send_telegram_notification(f"🔻 [매도 신호] {self.symbol} ({interval})")
                            self.notification_tracker[signal_key] = time.time()

                processed_data_package[interval] = processed_data

            # 3. 전역 캐시에 데이터 업데이트 (Lock 사용으로 스레드 안전성 확보)
            with lock:
                app_data_cache[self.symbol] = processed_data_package
        
            time.sleep(1) # 1초마다 실시간 가격 반영

# --- ✨✨✨ 핵심 수정: 멀티스레드 시작 함수 ---
def start_data_workers():
    """COIN_LIST에 있는 각 코인에 대해 별도의 워커 스레드를 시작합니다."""
    print("Starting data worker threads...")
    for symbol in COIN_LIST:
        worker = CoinWorker(symbol=symbol, client=client)
        worker.start()
    print("All coin data workers have been started.")