import streamlit as st
import requests
import json

import pandas as pd
import numpy as np

import plotly.express as px

def run():
    st.header("Check the emotion in the Text.")

    option = st.selectbox('Algorithm you would like to use?',('LogisticRegression', 'Naive Bias'))
    input_feature = st.text_input("Enter the statment to predict the  emotion type", "I am happy to help you.")

    data = {
        'option' : option,
        'input_feature' : input_feature
    }

    if st.button('Predict'):
            response = requests.post(' http://127.0.0.1:8000/predict' , json = data)
            loaded_json = json.loads(response.text)

            emotion = np.array(loaded_json['emotion'])
            data = np.array(loaded_json['data'])

            df = pd.DataFrame()
            df['Emotion'] = data
            df['emotion_propb'] = emotion.T

            st.markdown("The Emotion Predectied in the text is: "+ loaded_json['prediction'])

            fig = px.pie(df['Emotion'] , values = df['emotion_propb'], names ="Emotion")

            st.write(fig)

if __name__ == '__main__':
    run()
