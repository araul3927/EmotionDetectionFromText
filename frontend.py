import streamlit as st
from api_call import api_call

import pandas as pd
import numpy as np

import plotly.express as px


def run():
    st.header("Check the emotion in the Text.")

    selected_model = st.selectbox(
        "Algorithm you would like to use?", ("Logistic Regression", "Naive Bias")
    )
    input_text = st.text_input(
        "Enter the statment to predict the  emotion type", "Enter your text here"
    )

    if st.button("Predict"):
        response = api_call(selected_model, input_text)

        emotion = np.array(response.get("emotion"))
        data = np.array(response.get("data"))

        df = pd.DataFrame()
        df["Emotion"] = data
        df["emotion_propb"] = emotion.T

        st.markdown(
            f'The Emotion Predectied in the text is: { response.get("prediction") } '
        )

        fig = px.pie(df["Emotion"], values=df["emotion_propb"], names="Emotion")

        st.write(fig)


if __name__ == "__main__":
    run()
