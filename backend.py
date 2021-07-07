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

    if data.model_name == "Logistic Regression":
        prediction = lr.predict(test_sentence)
        prediction_label = prediction[0]
        emotions = lr.predict_proba(test_sentence)
        datas = lr.classes_
    else:
        prediction = nb.predict(test_sentence)
        prediction_label = prediction[0]
        emotions = nb.predict_proba(test_sentence)
        datas = nb.classes_

    return {
        "prediction": prediction_label,
        "emotion": emotions.tolist(),
        "data": datas.tolist(),
    }


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", reload=True)
