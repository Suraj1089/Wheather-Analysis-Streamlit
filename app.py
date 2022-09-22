import streamlit as st
import pandas as pd
import numpy as np
import plost
from PIL import Image
import requests
from dotenv import load_dotenv
import os 

load_dotenv(".env")
WHEATHER_API_KEY=os.getenv('WHEATHER_API_KEY')
# Page setting
st.set_page_config(layout="wide")


@st.cache
def load_city_names():
    city_list = pd.read_csv('city_names.csv')
    return city_list

city_list = load_city_names()
st.title('Wheather data Analysis')
with open('css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Data
def show_wheather_details(city):
    res = requests.get('http://api.weatherapi.com/v1/current.json?key={0}&q={1}&aqi=no'.format(WHEATHER_API_KEY,city.lower()))
    res = dict(res.json())
    
    city = res['location']['name']
    country = res['location']['country']
    localtime = res['location']['localtime'].split()[1]
    
    temp = res['current']['temp_c']
    text = res['current']['condition']['text']
    icon = res['current']['condition']['icon']
    wind_speed = res['current']['wind_kph']
    wind_dir = res['current']['wind_dir']
    if wind_dir == 'W':
        wind_dir = 'West' 
    elif wind_dir == 'S':
        wind_dir = 'South'
    elif wind_dir == 'E':
        wind_dir = 'East' 
    elif wind_dir == 'N':
        wind_dir = 'North'
    humidity = res['current']['humidity']

    
    
    return city,country,localtime,temp,text,icon,wind_speed,wind_dir,humidity


city_name = st.selectbox(
    'Enter city Name',
    city_list['city'].values
)
button = st.button('Show')
if city_name and button:
    try:
        city,country,localtime,temp,text,icon,wind_speed,wind_dir,humidity = show_wheather_details(city_name)

        seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
        stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

        # Row A
        a1, a2, a3,a4 = st.columns(4)
        a1.markdown(
            """
            <a href="https://docs.streamlit.io">
                <img src={} />
            </a>""".format(icon),
            unsafe_allow_html=True
        )
        a2.metric("City", city, "")
        a3.metric("Country", country, "")
        a4.metric('Time',localtime,'')

        # Row Bw
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Temperature", "{} °C".format(temp))
        b2.metric("Wind", "{} kph".format(wind_speed), "-8%")
        b3.metric("Humidity", "{}%".format(humidity), "4%")
        b4.metric("Wind direction", wind_dir, "")

        # Row C
        c1, c2 = st.columns((7,3))
        with c1:
            st.markdown('### Heatmap')
            plost.time_hist(
            data=seattle_weather,
            date='date',
            x_unit='week',
            y_unit='day',
            color='temp_max',
            aggregate='median',
            legend=None)
        with c2:
            st.markdown('### Bar chart')
            plost.donut_chart(
                data=stocks,
                theta='q2',
                color='company')
    except KeyError:
        st.write('Please Enter a valid city Name or Not able to find wheather for entered location')

