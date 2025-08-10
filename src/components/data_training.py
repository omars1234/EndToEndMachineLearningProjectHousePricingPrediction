import joblib
from src.config.configuration import DataTrainingConfig
import pandas as pd
import os
from constants.main import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


class DataTraining:
    def __init__(self,data_training_config:DataTrainingConfig=DataTrainingConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        self.data_training_config = data_training_config
 
        
    def trainig(self):
       
        df=pd.read_csv(self.data_training_config.cleaned_data_file_dir)

        x=df.drop(self.data_training_config.target_feature,axis=1)
        y=df[self.data_training_config.target_feature] 

        preprocessor = joblib.load(self.data_training_config.preprocessor_dir) 
       
        preprocessed_data=preprocessor.transform(x)
                        

        X_train, X_val, y_train, y_val = train_test_split(preprocessed_data, y,
                                                          test_size=self.data_training_config.test_split_ratio,
                                                          random_state=self.data_training_config.random_state)

        model=GradientBoostingRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(self.data_training_config.model_dir,MODEL_DIR_NAME))
        return X_train, X_val, y_train, y_val
        