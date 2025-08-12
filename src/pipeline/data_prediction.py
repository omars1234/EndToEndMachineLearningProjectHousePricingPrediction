import joblib
from pathlib import Path
import numpy as np
from constants.main import *
import pandas as pd
from requests import request

class PredictionPipeline:
    def __init__(self):
        self.preprocessor = joblib.load(os.path.join(self.data_transformation_config.preproccesor_transformation_object,PREPROCCESOR_TRANSFORMATION_OBJECt_DIR_NAME))
        self.model=joblib.load(os.path.join(self.data_training_config.model_dir,MODEL_DIR_NAME))

    def predict(self,input_data):
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        #preprocessed_data=self.preprocessor.transform(input_df)
        prediction=self.model.predict(input_df)

        return np.round(prediction)