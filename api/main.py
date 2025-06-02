from fastapi import FastAPI
from api.schemas import CarFeatures
from api.model_loader import model, preprocess_input

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Car Price Prediction API is running"}

@app.post("/predict")
def predict(features: CarFeatures):
    input_dict = features.dict()  # or features.model_dump() if pydantic v2
    X_scaled = preprocess_input(input_dict)
    prediction = model.predict(X_scaled)
    return {"predicted_price": round(prediction[0], 2)}
