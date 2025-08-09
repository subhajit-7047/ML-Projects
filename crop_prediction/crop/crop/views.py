from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
import pickle
import os

model_path = os.path.join(settings.BASE_DIR, "decesionTree.pkl") 

with open(model_path, "rb") as file:
    model = pickle.load(file)

def home(request):
    pred=None
    if request.method =="POST":   
        temperature = request.POST['temperature']
        humidity =request.POST['humidity']
        ph=request.POST['ph']
        water=request.POST['water']
        season=request.POST['season']
       
        pred = model.predict([[temperature,humidity,ph,water,season]])
        #print(pred)
    return render(request, 'index.html', {"prediction":pred})