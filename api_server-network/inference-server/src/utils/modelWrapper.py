import mlflow.pyfunc
import joblib
import pandas as pd

class SarimaxWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # 저장된 SARIMAXResultsWrapper 로드
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        if isinstance(model_input, dict):
            steps = model_input.get("steps", 24)
            exog = model_input.get("exog", None)
        elif isinstance(model_input, pd.DataFrame):
            steps = int(model_input.iloc[0]["steps"])
            exog = None
        else:
            raise ValueError("model_input must be dict or pd.DataFrame")

        forecast = self.model.forecast(steps=steps, exog=exog)
        return forecast

class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        forecast = self.model.predict(model_input)
        return forecast