import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Emotion Analytics Dashboard", layout="wide")

st.title("🧠 Emotion Detection Analytics")

uploaded_file = "results/emotion_results.csv"
df = pd.read_csv(uploaded_file)

st.dataframe(df.tail(10), use_container_width=True)

# Count plot
fig1 = px.histogram(df, x="Dominant_Emotion", title="Frequency of Detected Emotions", color="Dominant_Emotion")
st.plotly_chart(fig1, use_container_width=True)

# Confidence trend
fig2 = px.line(df, x="Timestamp", y="Confidence", title="Confidence Trend Over Time", markers=True)
st.plotly_chart(fig2, use_container_width=True)

# Emotion distribution pie
fig3 = px.pie(df, names="Dominant_Emotion", title="Overall Emotion Distribution")
st.plotly_chart(fig3, use_container_width=True)
