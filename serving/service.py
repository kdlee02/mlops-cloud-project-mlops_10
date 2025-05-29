from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path

app = FastAPI()

# Path to the predicted CSV
CSV_PATH = Path("your/dataset/dir/future_temperature.csv")  # Update this path accordingly

@app.get("/forecast")
def get_forecast():
    if not CSV_PATH.exists():
        return JSONResponse(status_code=404, content={"error": "Prediction file not found"})
    df = pd.read_csv(CSV_PATH)
    return df.to_dict(orient="records")