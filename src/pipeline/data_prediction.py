import joblib
from pathlib import Path
import numpy as np
from constants.main import *

class PredictionPipeline:
    def __init__(self):
        self.preprocessor = joblib.load(os.path.join(self.data_transformation_config.preproccesor_transformation_object,PREPROCCESOR_TRANSFORMATION_OBJECt_DIR_NAME))
        self.model=joblib.load(os.path.join(self.data_training_config.model_dir,MODEL_DIR_NAME))

    def predict(self,input_data):
        preprocessed_data=self.preprocessor.transform(input_data)
        prediction=self.model.predict(preprocessed_data)

        return np.round(prediction)