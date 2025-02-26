import streamlit as st
import pandas as pd
import numpy as np


st.title('🤖 Machine Learning App 🤖')

st.info('This app will help you predict the pEC50 of apelin receptor agonists.')

st.write('This data is used to build my model')
with st.expander('Data'):
  st.write('###Standardized data Apelin')
  df = pd.read_csv('https://raw.githubusercontent.com/LamDuy77777/data/refs/heads/main/Apelin_1715.csv')
  df

with st.expander('Data visualization'):
  st.write("### Distribution of pEC50")

#Data preparations
with st.sidebar: 
  st.header('Input features')
