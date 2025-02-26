import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
st.title('🤖 Machine Learning App 🤖')

st.info('This app will help you predict the pEC50 of apelin receptor agonists.')

st.write('This data is used to build my model')
with st.expander('Data'):
  st.write('**Standardized data Apelin**')
  df = pd.read_csv('https://raw.githubusercontent.com/LamDuy77777/data/refs/heads/main/Apelin_1715.csv')
  df

with st.expander('Data visualization'):
  st.write("### Distribution of pEC50")
  fig = px.histogram(df, x='pEC50', nbins=15, title="Distribution of pEC50", color_discrete_sequence=['skyblue'])
  fig.update_layout(xaxis_title="pEC50", yaxis_title="Frequency")
  st.plotly_chart(fig)

