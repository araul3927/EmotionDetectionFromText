# Import Needed Libraries
import joblib
import uvicorn
from pydantic import BaseModel

from fastapi import FastAPI


app = FastAPI(title="NLP API")

# load the models
lr = joblib.load("model/lr_model.joblib")
nb = joblib.load("model/nb_model.joblib")
tfidf = joblib.load("model/tfidf_model.joblib")


class Data(BaseModel):
    model_name: str
    input_text: str


# Api home endpoint
@app.get("/")
def home():
    return {"message": "API running"}


# NLP API end point
@app.post("/predict")
def predict(data: Data):
    test_sentence = tfidf.transform([data.input_text])

    if data.model_name == "lr":
        prediction = lr.predict(test_sentence)
        prediction_label = prediction[0]
    else:
        prediction = nb.predict(test_sentence)
        prediction_label = prediction[0]

    return {"prediction": prediction_label}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
