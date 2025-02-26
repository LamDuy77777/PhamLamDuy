import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('🤖 Machine Learning App 🤖')

st.info('This app will help you predict the pEC50 of apelin receptor agonists.')

st.write('This data is used to build my model')
with st.expander('Data'):
  st.write('**Standardized data Apelin**')
  df = pd.read_csv('https://raw.githubusercontent.com/LamDuy77777/data/refs/heads/main/Apelin_1715.csv')
  df

with st.expander('Data visualization'):
  st.write("### Distribution of pEC50")
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.histplot(data=df, x='pEC50', bins=15, kde=False, color='skyblue', edgecolor='black', ax=ax)
  ax.set_xlabel("pEC50")
  ax.set_ylabel("Frequency")
  ax.set_title("Distribution of pEC50")

