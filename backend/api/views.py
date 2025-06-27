# backend/api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from . import data_manager
import pandas as pd
import numpy as np # NaN 값을 처리하기 위해 numpy 임포트

class ChartDataView(APIView):
    def get(self, request, symbol):
        # data_manager에 있는 전역 캐시에서 데이터를 가져옴
        with data_manager.lock:
            data = data_manager.app_data_cache.get(symbol, {})
        
        payload = {}
        for tf, tf_data in data.items():
            if tf_data and isinstance(tf_data.get('plot_df'), pd.DataFrame) and not tf_data['plot_df'].empty:
                
                # --- ✨✨✨ 핵심 수정: NaN 값을 null로 변환하는 로직 추가 ✨✨✨ ---

                # 1. 캔들 데이터 변환
                plot_df_json = tf_data['plot_df'].reset_index()
                plot_df_json['Date'] = plot_df_json['Date'].apply(lambda x: int(x.timestamp()))
                
                # 2. 스토캐스틱 RSI 데이터 변환
                stoch_rsi_json = tf_data['stoch_rsi'].reset_index()
                stoch_rsi_json['Date'] = stoch_rsi_json['Date'].apply(lambda x: int(x.timestamp()))

                # 3. 커널 RSI 데이터 변환
                kernel_rsi_json = tf_data['kernel_rsi'].reset_index()
                kernel_rsi_json.columns = ['Date', 'Value']
                kernel_rsi_json['Date'] = kernel_rsi_json['Date'].apply(lambda x: int(x.timestamp()))
                
                # 4. 신호 데이터 변환
                stoch_buy_json = tf_data['stoch_buy'][tf_data['stoch_buy']].index.to_series().apply(lambda x: int(x.timestamp())).tolist()
                stoch_sell_json = tf_data['stoch_sell'][tf_data['stoch_sell']].index.to_series().apply(lambda x: int(x.timestamp())).tolist()
                krsi_long_json = tf_data['krsi_long'][tf_data['krsi_long']].index.to_series().apply(lambda x: int(x.timestamp())).tolist()
                krsi_short_json = tf_data['krsi_short'][tf_data['krsi_short']].index.to_series().apply(lambda x: int(x.timestamp())).tolist()
                
                # 최종 payload에 가공된 데이터 담기 (NaN을 None으로 변환 후 dict로 변환)
                payload[tf] = {
                    "candles": plot_df_json.replace({np.nan: None}).to_dict('records'),
                    "stoch_rsi": stoch_rsi_json.replace({np.nan: None}).to_dict('records'),
                    "kernel_rsi": kernel_rsi_json.replace({np.nan: None}).to_dict('records'),
                    "stoch_buy": stoch_buy_json,
                    "stoch_sell": stoch_sell_json,
                    "krsi_long": krsi_long_json,
                    "krsi_short": krsi_short_json,
                }
        
        return Response(payload)