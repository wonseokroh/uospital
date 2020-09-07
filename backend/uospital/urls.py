"""uospital URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from recommendation_sys import views as recommendation_sys_views

urlpatterns = [
    path(
        'admin/',
        admin.site.urls
    ),
    path(
        'recommendation/',
        recommendation_sys_views.recommendation_page,
        name='recommendation'
    ),
    path(
        'recommendation/content/',
        recommendation_sys_views.recommendation_by_content,
        name='recommendation-by-content'
    ),
    path(
        'recommendation/collaborative/',
        recommendation_sys_views.recommendation_by_collaborative,
        name='recommendation-by-collaborative'
    )
    path(
        'recommendation/neural/',
        recommendation_sys_views.recommendation_by_neural_collabo,
        name='recommendation-by-neural'
    )
]
