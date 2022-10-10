from random import sample
import streamlit as st
import pandas as pd
import numpy as np
import plost
from PIL import Image
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt 
import os 
import plotly.express as px 

load_dotenv(".env")
# Page setting


def extract_date_time(data,column_name):
    df = data 
    df[column_name] = pd.to_datetime(df[column_name], utc=True)
    df[column_name + '_year'] = df[column_name].dt.year
    df[column_name + '_month'] = df[column_name].dt.month
    df[column_name + '_day'] = df[column_name].dt.day
    df[column_name + '_hour'] = df[column_name].dt.hour
    return df.drop(column_name, axis=1)

def plot_scatter(x,y):
    fig = px.scatter(x=x, y=y)
    return fig

 

def plot_bar(x,y):

    fig = px.bar(x=x, y=y)
    return fig


def plot_line(x,y):
    try:
        fig = px.line(x=x, y=y)
        return fig
    except:
        st.error(f'cant able to plot line plot for {x} and {y}')
        return

def plot_pie(x,y):
    try:
        fig = px.pie(x=x, y=y)
        return fig         
    except:
        st.error(f'cant able to plot pie plot for {x} and {y}')
        return

if __name__ == '__main__':
        
    st.set_page_config(layout="wide",page_icon="⛅")


    #hide navbar menu and streamlit icon

    hide_menu_icon = st.markdown(
        """
        <style>
            #MainMenu {visibility : hidden;}
            footer {visibility : hidden;}

        """,
        unsafe_allow_html=True
    )

    st.title('Wheather data Analysis')
    with open('css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



    # Sidebar
    st.sidebar.title('Wheather data Analysis')
    file = st.sidebar.file_uploader('Upload your file', type=['csv','excel'])
    sample_data = st.checkbox('See Sample Dashboard',key='sample_data')
    sample_file = None 
    if sample_data:
        sample_file = 'data/weatherHistory.csv'
    if sample_file is not None:
        df = pd.read_csv(sample_file)
        df.dropna(inplace=True)
        with st.expander('See raw data'):
            st.write(df)
        df = extract_date_time(df,'Formatted Date')
        st.write(df)
            
       

