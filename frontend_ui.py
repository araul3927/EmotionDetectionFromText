import streamlit as st
from api_call import api_call

st.title("Emotion Detection")
st.write("""
Joy 😂, Fear 😨, Anger 😠, Sad 😟, Disgust 🤢, Shame 😳, Guilt 😓
""")

option = st.sidebar.selectbox(
    label='Which number do you like best?',
    options=("Logistic Regression","Multinominal NB"))

st.markdown(f'You seleted: **{option}** ')

txt = st.text_area('Text to analyze',)

if st.button('Predict'):
    emotion = api_call(option,txt)
    st.write(f'Emotion: { emotion.get("prediction") }')