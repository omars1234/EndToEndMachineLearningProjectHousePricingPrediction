from src.entity.config_entity import DataIngestionConfig
import pandas as pd
import os
from constants.main import *


class DataIngestion:
    def __init__(self,ingestion_config:DataIngestionConfig=DataIngestionConfig()):

        self.ingestion_config = ingestion_config
 
        
    def export_data_into_feature_store(self):
       
        df=pd.read_csv(self.ingestion_config.data_file_path_dir)
        df.to_csv(os.path.join(self.ingestion_config.data_ingested_dir,RAW_DATA_FILE_PATH_DIR_NAME),index=False,header=True)