from django.urls import path
from . import views

# Main URL patterns for the recommendation system
urlpatterns = [
    # Home page - main recommendation interface
    path('', views.home, name='home'),

    # Alternative paths (optional - you can remove these if not needed)
    path('recommendations/', views.home, name='recommendations'),
    path('search/', views.home, name='search'),
]