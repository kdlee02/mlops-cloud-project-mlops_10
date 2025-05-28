import bentoml
import pandas as pd
from bentoml.io import JSON

runner = bentoml.sklearn.get("weather_predictor:latest").to_runner()
svc = bentoml.Service("weather_prediction_service", runners=[runner])

@svc.api(input=JSON(), output=JSON())
def predict(input_json:dict) -> dict:
    try:
        if not isinstance(input_json, dict):
            raise ValueError("Input must be a dictionary")
        
        df = pd.DataFrame([input_json])
        prediction = runner.run(df)
        return {"prediction": prediction[0]}  
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}