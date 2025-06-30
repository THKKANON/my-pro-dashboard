# backend/api/data_manager.py
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

TELEGRAM_BOT_TOKEN = "여기에_텔레그램_봇_토큰을_붙여넣으세요"
TELEGRAM_CHAT_ID = "여기에_텔레그램_채팅_ID를_붙여넣으세요"

client = Client()

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
        if i == 0: ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (df.iloc[i]['Open'] + df.iloc[i]['Close']) / 2
        else: ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (ha_df.iloc[i-1]['Open'] + ha_df.iloc[i-1]['Close']) / 2
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
    except (ValueError, IndexError): return 0
    return 0

# ✨✨✨ 핵심 수정: process_dataframe 함수 이름 변경 및 역할 명확화 ✨✨✨
def calculate_indicators(df):
    """주어진 DataFrame에 대해 모든 보조지표와 차트 데이터를 계산하여 하나의 딕셔너리로 반환합니다."""
    if df is None or df.empty:
        return None

    # 모든 계산은 원본 df의 복사본으로 수행
    df_copy = df.copy()

    stoch_rsi = df_copy.ta.stochrsi()
    if stoch_rsi is None or stoch_rsi.empty: return None

    stoch_buy, stoch_sell = find_stoch_rsi_signals(stoch_rsi)

    rsi = df_copy.ta.rsi(length=KERNEL_RSI_LENGTH)
    if rsi is None or rsi.dropna().empty: return None

    kernel_rsi_values = kernel_regression(rsi.dropna().values, KERNEL_RSI_BANDWIDTH)
    kernel_rsi = pd.Series(kernel_rsi_values, index=rsi.dropna().index)

    krsi_long, krsi_short = find_kernel_rsi_signals(kernel_rsi)

    # 캔들차트(plot_df) 계산도 이 함수 안에서 함께 처리
    plot_df = calculate_heikin_ashi(df_copy) if CHART_TYPE == 'Heikin-Ashi' else df_copy

    return {
        'plot_df': plot_df,
        'stoch_rsi': stoch_rsi,
        'stoch_buy': stoch_buy,
        'stoch_sell': stoch_sell,
        'kernel_rsi': kernel_rsi,
        'krsi_long': krsi_long,
        'krsi_short': krsi_short
    }

class CoinWorker(threading.Thread):
    def __init__(self, symbol, client):
        super().__init__()
        self.symbol = symbol
        self.client = client
        self.daemon = True
        self.kline_data_cache = {}
        self.last_kline_fetch_time = 0
        self.indicator_thread = threading.Thread(target=self._indicator_calculation_loop, daemon=True)
        self.new_data_event = threading.Event()

    def _indicator_calculation_loop(self):
        """백그라운드에서 모든 지표 계산을 담당하고, 전역 캐시를 원자적으로 업데이트합니다."""
        print(f"[{time.strftime('%H:%M:%S')}] {self.symbol}: Indicator calculation thread started.")
        while True:
            self.new_data_event.wait(timeout=10)
            if self.new_data_event.is_set():
                # ✨ 1. 이 스레드에서 사용할 데이터만 지역 변수로 복사 (동시성 문제 방지)
                local_kline_cache = self.kline_data_cache.copy()

                for interval in TIMEFRAME_OPTIONS:
                    df = local_kline_cache.get(interval)
                    if df is None:
                        continue
                    
                    # ✨ 2. 캔들차트와 모든 보조지표를 한 번에 계산
                    # 계산된 결과는 완전히 일관성이 보장됨
                    complete_data_package = calculate_indicators(df)
                    
                    if complete_data_package:
                        # ✨ 3. Lock을 걸고 전역 캐시를 통째로 교체 (업데이트가 아닌 교체)
                        # 이렇게 하면 데이터 불일치 상태가 발생하지 않음
                        with lock:
                            if self.symbol not in app_data_cache:
                                app_data_cache[self.symbol] = {}
                            app_data_cache[self.symbol][interval] = complete_data_package

                self.new_data_event.clear()
            time.sleep(0.1)

    def run(self):
        """이 스레드는 가격 데이터를 가져와 내부 캐시(`kline_data_cache`)를 업데이트하고,
           계산 스레드에 신호를 보내는 역할만 합니다."""
        print(f"[{time.strftime('%H:%M:%S')}] Price update worker started for {self.symbol}")
        self.indicator_thread.start()

        while True:
            now = time.time()
            data_updated = False

            if now - self.last_kline_fetch_time > KLINE_REFRESH_INTERVAL_SECONDS:
                print(f"[{time.strftime('%H:%M:%S')}] {self.symbol}: Performing full kline data refresh...")
                for interval in TIMEFRAME_OPTIONS:
                    # ✨ 4. 전역 캐시가 아닌, 이 인스턴스에만 속한 `kline_data_cache`를 업데이트
                    self.kline_data_cache[interval] = get_binance_futures_candles(self.symbol, interval, FETCH_COUNT)
                    time.sleep(0.05)
                self.last_kline_fetch_time = now
                data_updated = True
                print(f"[{time.strftime('%H:%M:%S')}] {self.symbol}: Full kline data refreshed.")

            try:
                ticker = self.client.futures_ticker(symbol=self.symbol)
                latest_price = float(ticker['lastPrice'])
            except Exception as e:
                print(f"Error fetching ticker for {self.symbol}: {e}")
                time.sleep(1)
                continue

            for interval in TIMEFRAME_OPTIONS:
                df = self.kline_data_cache.get(interval)
                if df is None or df.empty: continue
                
                df_copy = df.copy()
                last_candle_time = df_copy.index[-1]
                interval_seconds = get_interval_seconds(interval)
                if interval_seconds == 0: continue
                
                now_kst = datetime.fromtimestamp(now, tz=timezone(timedelta(hours=9)))
                
                if now_kst >= last_candle_time + timedelta(seconds=interval_seconds):
                    new_candle_open_time = last_candle_time + timedelta(seconds=interval_seconds)
                    new_row_data = {'Open': df_copy.iloc[-1]['Close'], 'High': latest_price, 'Low': latest_price, 'Close': latest_price}
                    new_row = pd.DataFrame([new_row_data], index=[new_candle_open_time])
                    new_row.index.name = 'Date'
                    df_copy = pd.concat([df_copy, new_row]).iloc[1:]
                    self.kline_data_cache[interval] = df_copy
                    data_updated = True
                else:
                    df_copy.iloc[-1, df_copy.columns.get_loc('Close')] = latest_price
                    df_copy.iloc[-1, df_copy.columns.get_loc('High')] = max(df_copy.iloc[-1]['High'], latest_price)
                    df_copy.iloc[-1, df_copy.columns.get_loc('Low')] = min(df_copy.iloc[-1]['Low'], latest_price)
                    self.kline_data_cache[interval] = df_copy
                    data_updated = True

            # ✨ 5. 데이터에 변경이 있을 때만 계산 스레드에 신호를 보냄
            if data_updated:
                self.new_data_event.set()

            time.sleep(1)

def start_data_workers():
    print("Starting data worker threads...")
    for symbol in COIN_LIST:
        worker = CoinWorker(symbol=symbol, client=client)
        worker.start()
    print("All coin data workers have been started.")