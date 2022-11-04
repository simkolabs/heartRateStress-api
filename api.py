from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from src.utils import Model
import os




app = FastAPI()
#root
@app.get("/")
def root():
    return "api running!"


#prediction endpoint
@app.post("/predict",responses={200: {"description": "Prediction CSV", "content" : {"file/csv" : {"example" : "No example available. csv file"}}}})
async def predict(file: UploadFile = File(...)):
    model=Model()
    response=model.predict(file=file)
    file_path = os.path.join("predictions/predictions.csv")
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="predictions.csv")
    return {"error" : "File not found!"}