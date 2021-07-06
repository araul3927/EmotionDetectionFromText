import urllib3
import json
import ast

ENDPOINT = "http://127.0.0.1:8000/predict"
http = urllib3.PoolManager()


def api_call(option, input_text):

    """Returns the predicted emotion for the text using the choosen model

    Parameters:
    option(string) : Model to use
    input_text(string) : The input text for emotion prediction
    Returns:
    data(dictionary) : The prediction emotion
    """

    if option == "Logistic Regression":
        model_name = "lr"
    else:
        model_name = "nb"

    model_and_text = json.dumps({"model_name": model_name, "input_text": input_text})

    headers = {"Content-Type": "application/json"}
    response = http.request("POST", ENDPOINT, headers=headers, body=model_and_text)
    bytes_data = response.data
    data = ast.literal_eval(bytes_data.decode("utf-8"))

    return data
