# urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # 'api/'로 시작하는 모든 요청을 api.urls로 전달
    path('api/', include('api.urls')),
]