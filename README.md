# EmotionDetectionFromText
This project tries to predict the emotions of a text document.


## Installation

To run this project you would need:

- Download/ Clone the project

```git
  git clone https://github.com/araul3927/EmotionDetectionFromText.git
```

- Create a virtual environment

```python3
  python3 -m venv env
```

- Activate the environment
```bash
  source env/bin/activate
```

- Install the required packages

```python3
  pip3 install -r requirements.txt

``` 

- Run the UI to know about data and its accuracy
```python3
  streamlit run model_train.py
```

- Run the server
```python3
  python3 backend.py

```

- Run the UI to predict the emotion of text
```python3
  streamlit run frontend.py
```

If you want to see behind the scenes regarding the model, we have included the ipynb file.

