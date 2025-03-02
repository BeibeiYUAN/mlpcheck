from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

app = FastAPI()

# Load the model from MLflow
model = mlflow.sklearn.load_model("mlruns/798064658889095052/6bdd1d18a39242dcbe7557787ab56206/artifacts/random_forest_model")

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
