import joblib
from pathlib import Path
import numpy as np
from constants.main import *
import pandas as pd
from requests import request

class PredictionPipeline:
    def __init__(self):
        self.preprocessor = joblib.load(Path('saved_preproccesor\preprocessor.joblib'))
        self.model=joblib.load(Path('saved_model\model.joblib'))

    def predict(self,input_data):
        preprocessed_data=self.preprocessor.transform(input_data)
        prediction=self.model.predict(preprocessed_data)

        return np.round(prediction)