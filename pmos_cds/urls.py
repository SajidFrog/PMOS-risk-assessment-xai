from django.urls import path
from pmos_cds import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/assess/", views.assess_risk, name="assess_risk"),
    path("api/history/", views.symptom_history, name="symptom_history"),
]