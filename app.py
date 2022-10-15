from random import sample
import streamlit as st
import pandas as pd
import numpy as np
import plost
import seaborn as sns
from PIL import Image
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt 
import os 
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import plotly.express as px 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

load_dotenv(".env")
# Page setting

@st.cache(allow_output_mutation=True)
def impute_missing_values(df):
    imputer_int = KNNImputer(missing_values=np.nan)
    df['Sunshine'] = imputer_int.fit_transform(df[['Sunshine']])
    imputer_str = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    df['WindGustDir'] = imputer_str.fit_transform(df[['WindGustDir']])
    df['WindGustSpeed'] = imputer_int.fit_transform(df[['WindGustSpeed']])
    df['WindDir9am'] = imputer_str.fit_transform(df[['WindDir9am']])
    df['WindDir3pm'] = imputer_str.fit_transform(df[['WindDir3pm']])
    df['WindSpeed9am'] = imputer_int.fit_transform(df[['WindSpeed9am']])
    df.drop('RISK_MM', inplace=True,axis=1)
    return df

def extract_date_time(data,column_name):
    df = data 
    df[column_name] = pd.to_datetime(df[column_name], utc=True)
    df[column_name + '_year'] = df[column_name].dt.year
    df[column_name + '_month'] = df[column_name].dt.month
    df[column_name + '_day'] = df[column_name].dt.day
    df[column_name + '_hour'] = df[column_name].dt.hour
    return df.drop(column_name, axis=1)


if __name__ == '__main__':
        
    st.set_page_config(layout="wide",page_icon="⛅")


    #hide navbar menu and streamlit icon

    # hide_menu_icon = st.markdown(
    #     """
    #     <style>
    #         #MainMenu {visibility : hidden;}
    #         footer {visibility : hidden;}

    #     """,
    #     unsafe_allow_html=True
    # )

    st.title('Wheather data Analysis')
    with open('css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



    # Sidebar
    st.title('Wheather data Analysis')
    df = pd.read_csv('data/weather.csv')
    if df is not None:
        df.dropna(inplace=True)
        with st.expander('See raw data'):
            st.write(df)

        st.markdown('## Data Visualization')
        df = impute_missing_values(df)
        c1,c2 = st.columns(2)
        with c1:
            c1.metric('Minimum Temperature',df['MinTemp'].min())
            fig = px.scatter(df, x="MinTemp", y="Temp9am", color="Rainfall")
            st.plotly_chart(fig)
        with c2:
            c2.metric('Maximum Temperature',df['MaxTemp'].max())
            fig = px.imshow(df.corr())
            st.plotly_chart(fig)
        
        c3,c4 = st.columns(2)
        with c3:
            st.warning('The graph shows the relation between Sunshine and Temperature at 3pm')
            fig = px.scatter(df, x="Sunshine", y="Temp3pm", color="Rainfall")
            st.plotly_chart(fig)
        
        with c4:
            st.warning('Wind speed at 3pm')
            fig = px.histogram(
                df['WindSpeed3pm'],
            )
            st.plotly_chart(fig)

        
            
        c5,c6 = st.columns(2)
        with c5:
            st.warning('Wind speed at 9pm')
            fig = px.histogram(
                df['WindSpeed9am'],
                )
            st.plotly_chart(fig)

        with c6:
            st.warning('Wind speed at 3pm')
            fig = px.histogram(
                df['WindSpeed3pm'],
            )
            st.plotly_chart(fig)

        


        LB = LabelBinarizer()
        df['WindGustDir'] = LB.fit_transform(df[['WindGustDir']])
        df['WindDir9am'] = LB.fit_transform(df[['WindDir9am']])
        df['WindDir3pm'] = LB.fit_transform(df[['WindDir3pm']])

        LE = LabelEncoder()
        df['RainToday'] = LE.fit_transform(df['RainToday'])
        df['RainTomorrow'] = LE.fit_transform(df['RainTomorrow'])
        df['RainTomorrow'] = LE.fit_transform(df['RainTomorrow'])
        X = df.drop('RainTomorrow',axis=1).values
        y = df['RainTomorrow'].values

        minmax = MinMaxScaler()
        X = minmax.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

        # prediction using decision tree 
        dtree = DecisionTreeClassifier()
        dtree.fit(X_train,y_train)
        previsor_dtree = dtree.predict(X_test)

        # prediction using random forest
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(X_train,y_train)
        previsor_rfc = rfc.predict(X_test)

        # prediction using support vector machine
        svc = SVC()
        svc.fit(X_train,y_train)
        previsor_svc = svc.predict(X_test)

        # prediction using logistic regression
        lr = LogisticRegression()
        lr.fit(X_train,y_train)
        predict_LR = lr.predict(X_test)


        Acuracia_LR = np.round(accuracy_score(y_test,predict_LR),3) *100
        Acuracia_svc = np.round(accuracy_score(y_test,previsor_svc),3)*100
        Acuracia_rfc = np.round(accuracy_score(y_test,previsor_rfc),3)*100
        Acuracia_dtree = np.round(accuracy_score(y_test,previsor_dtree),3)*100


        st.write('## Model Evaluation')
        st.write('### Accuracy')

        a1,a2,a3,a4 = st.columns(4)
        with a1:
            st.metric('Accuracy of Logistic Regression is',round(Acuracia_LR,2),'%')

        with a2:
            st.metric('Accuracy of Support Vector Machine is',round(Acuracia_svc,2),'%')
        
        with a3:
            st.metric('Accuracy of Random Forest is',round(Acuracia_rfc,2),'%')
        
        with a4:
            st.metric('Accuracy of Decision Tree is',round(Acuracia_dtree,2),'%')
        


        st.write('### Confusion Matrix')
        cm1,cm2 = st.columns(2)
        with cm1:
            st.write('Confusion Matrix of Logistic Regression')
            fig1 = px.imshow(confusion_matrix(y_test,predict_LR),labels=dict(x="Predicted", y="Actual", color="Count"),color_continuous_scale='Blues')
            st.plotly_chart(fig1)
        with cm2:
            st.write('Confusion Matrix of Support Vector Machine')
            fig2 = px.imshow(confusion_matrix(y_test,previsor_svc),labels=dict(x="Predicted", y="Actual", color="Count"),color_continuous_scale='Blues')
            st.plotly_chart(fig2)
        
        cm3,cm4 = st.columns(2)
        with cm3:
            st.write('Confusion Matrix of Random Forest')
            fig3 = px.imshow(confusion_matrix(y_test,previsor_rfc),labels=dict(x="Predicted", y="Actual", color="Count"),color_continuous_scale='Blues')
            st.plotly_chart(fig3)
        
        with cm4:
            st.write('Confusion Matrix of Decision Tree')
            fig4 = px.imshow(confusion_matrix(y_test,previsor_dtree),labels=dict(x="Predicted", y="Actual", color="Count"),color_continuous_scale='Blues')
            st.plotly_chart(fig4)
        

        with st.form('Preicting Rain Tomorrow'):
            st.write('Enter the values to predict whether it will rain tomorrow or not')
            p1,p2,p3,p4,p5,p6,p7 = st.columns(7)
            with p1:
                MinTemp = st.number_input('Minimum Temperature',min_value=0.0,max_value=100.0,value=0.0)
            with p2:
                MaxTemp = st.number_input('Maximum Temperature',min_value=0.0,max_value=100.0,value=0.0)
            with p3:
                Rainfall = st.number_input('Rainfall',min_value=0.0,max_value=100.0,value=0.0)
            with p4:
                Evaporation = st.number_input('Evaporation',min_value=0.0,max_value=100.0,value=0.0)
            with p5:
                Sunshine = st.number_input('Sunshine',min_value=0.0,max_value=100.0,value=0.0)
            with p6:
                WindGustSpeed = st.number_input('Wind Gust Speed',min_value=0.0,max_value=100.0,value=0.0)
            with p7:
                WindSpeed9am = st.number_input('Wind Speed at 9am',min_value=0.0,max_value=100.0,value=0.0)
            
            p8,p9,p10,p11,p12,p13,p14 = st.columns(7)
            with p8:
                WindSpeed3pm = st.number_input('Wind Speed at 3pm',min_value=0.0,max_value=100.0,value=0.0)
            with p9:
                Humidity9am = st.number_input('Humidity at 9am',min_value=0.0,max_value=100.0,value=0.0)
            with p10:
                Humidity3pm = st.number_input('Humidity at 3pm',min_value=0.0,max_value=100.0,value=0.0)
            with p11:
                Pressure9am = st.number_input('Pressure at 9am',min_value=0.0,max_value=100.0,value=0.0)
            with p12:
                Pressure3pm = st.number_input('Pressure at 3pm',min_value=0.0,max_value=100.0,value=0.0)
            with p13:
                Cloud9am = st.number_input('Cloud at 9am',min_value=0.0,max_value=100.0,value=0.0)
            with p14:
                Cloud3pm = st.number_input('Cloud at 3pm',min_value=0.0,max_value=100.0,value=0.0)
            
            p15,p16,p17,p18,p19,p20,p21 = st.columns(7)
            with p15:
                Temp9am = st.number_input('Temperature at 9am',min_value=0.0,max_value=100.0,value=0.0)
            with p16:
                Temp3pm = st.number_input('Temperature at 3pm',min_value=0.0,max_value=100.0,value=0.0)
            with p17:
                RainToday = st.selectbox('Rain Today',('Yes','No'))
            with p18:
                WindGustDir = st.selectbox('Wind Gust Direction',('E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'))
            with p19:
                WindDir9am = st.selectbox('Wind Direction at 9am',('E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'))
            with p20:
                WindDir3pm = st.selectbox('Wind Direction at 3pm',('E','ENE','ESE','N','NE','NNE','NNW','NW','S','SE','SSE','SSW','SW','W','WNW','WSW'))
            with p21:
                Month = st.selectbox('Month',('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
            
        
            
            submitted = st.form_submit_button('Predict')
            if submitted:
                st.write('### Prediction')
                pred = lr.predict([[MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustSpeed,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm]])
                if pred == 0:
                    st.write('It will not rain tomorrow')
                else:
                    st.write('It will rain tomorrow')
            
