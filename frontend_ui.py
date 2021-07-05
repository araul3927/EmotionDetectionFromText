import streamlit as st
import time
import numpy as np

st.title("Emotion Detection")
st.write("""
Joy 😂, Fear 😨, Anger 😠, Sad 😟, Disgust 🤢, Shame 😳, Guilt 😓
""")

option = st.sidebar.selectbox(
    'Which number do you like best?',
     ("model1","model2"))

# st.write('You selected:', option)
st.markdown(f'You seleted: **{option}** ')

txt = st.text_area('Text to analyze',)

st.button('Predict')

st.write('Emotion:',)
