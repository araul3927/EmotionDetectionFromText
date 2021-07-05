import uvicorn
import joblib
import numpy as np
import pandas as pd

from pydantic import BaseModel

from fastapi import FastAPI

app = FastAPI()

tfidf=joblib.load('./model/tfidf_model.joblib')
lr=joblib.load('./model/lr_model.joblib')
nb=joblib.load('./model/nb_model.joblib')

class Data(BaseModel):
        option:str
        input_feature:str

@app.get('/')
def index():
    return{'key' : "API Testing"}

@app.post('/predict')
def predict(data:Data):
    test_sentence = tfidf.transform([data.input_feature])

    if(data.option == "Naive Bias"):        
        y = nb.predict(test_sentence)
        emotions = nb.predict_proba(test_sentence)
        datas = nb.classes_
    else:
        y = lr.predict(test_sentence)
        emotions = lr.predict_proba(test_sentence)
        datas = lr.classes_

    return{'prediction': y[0] , "emotion" : emotions.tolist() , 'data' : datas.tolist()}

if __name__=="__main__":
    uvicorn.run("main:app", reload = "True")