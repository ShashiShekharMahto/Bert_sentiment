from fastapi import FastAPI, Depends, File, UploadFile, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from typing import Dict, List

from Bert_sentiment_load_model import Model, get_model

app = FastAPI()
template = Jinja2Templates(directory='templates')

class Requests(BaseModel):
    text: str

class senti_Response(BaseModel):
    text : str
    #Ner : List = []
    senti : str

@app.get("/")
def home(request: Request):
    return template.TemplateResponse("index.html", {"request": request})


@app.post("/bert_sentiment", response_model=senti_Response)
def predict(request : Requests, model : Model = Depends(get_model)):
    probability = model.predict(request.text)
    print(probability)
    probability = "Positive" if probability > 0.46 else "Negative"
    return senti_Response(text = request.text,
                        senti = probability)