import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle 
import requests
import json
from datetime import date

st.write("""
# Air Quality Prediction App
This app predicts the **PM 2.5** Pollutant Levels in **New Delhi**
""")
st.write("[Live](https://rp5.ru/Weather_in_New_Delhi,_Safdarjung_(airport)) Weather Conditions for New Delhi")

st.sidebar.header('User Input Parameters')

# API details
API_KEY = ''
lat = '28.55'
lon = '77.10'
exclude = 'minute,alerts,daily'
units = 'metric'
# weather API call
url = 'https://api.openweathermap.org/data/2.5/onecall?lat='+lat+'&lon='+lon+'&exclude='+exclude+'&units='+units+'&appid='+API_KEY
#data = requests.get(url)
#data = data.json()

# pollution API call
url_pol = 'http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat='+lat+'&lon='+lon+'&appid='+API_KEY
#data_pol = requests.get(url_pol)
#data_pol = data_pol.json()

# user data/derived data
from datetime import date
today = date.today() 

def user_input_features():
    hr = st.sidebar.text_input('Hour (0-48)', 0)
    hour = int(hr, base=10)
    month = today.month

    data = requests.get(url)
    data = data.json()
    data_pol = requests.get(url_pol)
    data_pol = data_pol.json()

    # weather attributes
    temp = data['hourly'][hour]['temp']
    pres = data['hourly'][hour]['pressure']*0.75
    hum = data['hourly'][hour]['humidity']
    spd = data['hourly'][hour]['wind_speed']
    dir = data['hourly'][hour]['wind_deg']

    # pollution attributes
    pm10 =data_pol['list'][hour]['components']['pm10'] 
    no2 = data_pol['list'][hour]['components']['no2']
    nh3 = data_pol['list'][hour]['components']['nh3']
    no = data_pol['list'][hour]['components']['no']
    co = data_pol['list'][hour]['components']['co']/1000
    so2 = data_pol['list'][hour]['components']['so2']

    data = {
            'Temperature': temp,
            'Atmospheric_Pressure':pres, 
            'Relative_Humidity':hum,
            'Wind_Dir':dir, 
            'Wind_Speed':spd,
            'Month':month, 
            'Hour':hour, 
            'PM10':pm10, 
            'NO':no, 
            'NO2':no2,
            'NH3':nh3,
            'CO':co,
            'SO2':so2
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df[['Hour']])

st.subheader('Forecasted Parameters at that Time')
st.write(df)
data_pol = requests.get(url_pol)
data_pol = data_pol.json()
st.write(data_pol['list'][44]['components']['pm2_5'])
# building the model
with open("delhi_xgb.bin", 'rb') as f_in:
  model = pickle.load(f_in)

pred = model.predict(df)

st.subheader('Predicted PM 2.5 Level using our Model')
prediction = pred[0]
st.write(prediction)

st.subheader('Air Quality Index')
if(prediction >=0 and prediction <=12):
    st.write('Good')
elif(prediction >=12.1 and prediction <= 35.4):
    st.write('Moderate')
elif(prediction >=35.5 and prediction <= 55.4):
    st.write('Unhealty for Sensitive Groups')
elif(prediction >=55.5 and prediction <= 150.4):
    st.write('Unhealthy')
elif(prediction >=150.5 and prediction <= 250.4):
    st.write('Very Unhealthy')
else:
    st.write('Hazardous')    
    
st.subheader('Precautionary Measures')
if(prediction >=0 and prediction <=12):
    st.write('None needed!')
elif(prediction >=12.1 and prediction <= 35.4):
    st.write('Unusually sensitive people should consider reducing prolonged or heavy exertion.')
elif(prediction >=35.5 and prediction <= 55.4):
    st.write('People with respiratory or heart disease, the elderly and children should limit prolonged exertion.')
elif(prediction >=55.5 and prediction <= 150.4):
    st.write('People with respiratory or heart disease, the elderly and children should avoid prolonged exertion completely; everyone else should limit prolonged exertion.')
elif(prediction >=150.5 and prediction <= 250.4):
    st.write('People with respiratory or heart disease, the elderly and children should avoid any outdoor activity completely; everyone else should avoid prolonged exertion.')
else:
    st.write('Everyone should avoid any outdoor exertion; people with respiratory or heart disease, the elderly and children should remain indoors.')  
