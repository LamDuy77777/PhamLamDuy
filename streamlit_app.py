import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App 🤖')

st.info('This app will help you predict the pEC50 of apelin receptor agonists.')
df = pd.read_csv('https://github.com/LamDuy77777/data/blob/main/Apelin_1715.csv')
df
