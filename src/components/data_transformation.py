from src.config.configuration import DataTransformationConfig
import yaml
from constants.main import *
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler,PowerTransformer,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib



class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig=DataTransformationConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        self.data_transformation_config = data_transformation_config
        

        with open(self.data_transformation_config.schema_cleaned_file_path_dir) as yaml_file:
             claened_schema = yaml.safe_load(yaml_file)
        self.claened_schema_config=claened_schema
 
        
    def transform_df(self): 

        df=pd.read_csv(self.data_transformation_config.cleaned_data_file_dir)

        x=df.drop(TARGER_FEATURE,axis=1)
        y=df[TARGER_FEATURE]
        

        numeric_transformer = MinMaxScaler()
        oh_encoder=OneHotEncoder(sparse_output=False)

        num_transform_columns = self.claened_schema_config["num_features_for_transformation"]
        num_features = self.claened_schema_config["num_features_for_scaling"]
        cat_transform_columns=self.claened_schema_config["oh_columns"]


        transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))

            ])
        preprocessor = ColumnTransformer(
                            [   ("OneHotEncoder", oh_encoder, cat_transform_columns),
                                ("Transformer", transform_pipe, num_transform_columns),
                                ("StandardScaler", numeric_transformer, num_features),
                                
                            ]
                        )
        
        preprocessor=preprocessor.fit(x)
        
             
        joblib.dump(preprocessor, os.path.join(self.data_transformation_config.preproccesor_transformation_object,PREPROCCESOR_TRANSFORMATION_OBJECt_DIR_NAME))
        