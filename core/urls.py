from django.urls import path
from core import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('lesson/', views.lesson_detail, name='first_lesson'),  # redirects to first
    path('lessons/<int:lesson_id>/', views.lesson_detail, name='lesson_detail'),
    path('lessons/<int:lesson_id>/complete/', views.complete_lesson, name='complete_lesson'),
    path('process-image/', views.process_image, name='process_image'),
]
