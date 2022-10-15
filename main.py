from fastapi import FastAPI
from keytotext import trainer
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    keywords: Optional[list]= None

class model_select:
    model1= trainer()
    model1.load_model("model_bio", use_gpu=False)
    model2= trainer()
    model2.load_model("model", use_gpu=False)
    model3= trainer()
    model3.load_model("model_new", use_gpu=False)

app= FastAPI()

@app.post('/predict')
def return_bio(item:Item):
    output= []
    res1= model_select.model1.predict(item.keywords, use_gpu=False)
    res2= model_select.model2.predict(item.keywords, use_gpu=False)
    res3= model_select.model3.predict(item.keywords, use_gpu=False)
    output.append([res1, res2, res3])
    results= list(set(output[0]))
    return {"bio_generated":results}