import os
from constants.main import *
from src.entity.config_entity import (DataIngestionConfig,
                                      DataCleaningConfig,
                                      DataTransformationConfig)



class DataIngestionManager:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config

    def create_directories(self):
        os.makedirs(self.data_ingestion_config.data_artifacts_dir,exist_ok=True)
        os.makedirs(self.data_ingestion_config.data_ingested_dir,exist_ok=True) 



class DataCleaningManager:
    def __init__(self,cleaned_data_ingested_config:DataCleaningConfig=DataCleaningConfig()):
        self.cleaned_data_ingested_config=cleaned_data_ingested_config

    def create_directories(self):
        os.makedirs(self.cleaned_data_ingested_config.cleaned_data_ingested_dir,exist_ok=True)    



class DataTransformationManager:
    def __init__(self,data_transformation_config:DataTransformationConfig=DataTransformationConfig()):
        self.data_transformation_config=data_transformation_config

    def create_directories(self):
        os.makedirs(self.data_transformation_config.preproccesor_transformation_object,exist_ok=True)

