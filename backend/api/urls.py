# urls.py
from django.urls import path
from .views import ChartDataView

urlpatterns = [
    # api/chart-data/BTCUSDT/ 와 같은 요청을 ChartDataView로 연결
    path('chart-data/<str:symbol>/', ChartDataView.as_view(), name='chart-data'),
]