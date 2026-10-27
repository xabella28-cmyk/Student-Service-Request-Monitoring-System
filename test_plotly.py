import sys
import plotly
import plotly.express as px
import streamlit as st

st.title("Plotly Installation Test")
st.write("Python version:", sys.version)
st.write("Plotly version:", plotly.__version__)
st.write("Plotly location:", plotly.__file__)
st.plotly_chart(px.bar(x=["A", "B"], y=[1, 2]))
