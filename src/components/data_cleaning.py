
import os
import yaml
import pandas as pd
import numpy as np
from constants.main import *
from src.config.configuration import DataCleaningConfig


class DataCleaning:
    def __init__(self,data_cleaning_config:DataCleaningConfig=DataCleaningConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        self.data_cleaning_config = data_cleaning_config   
        with open(self.data_cleaning_config.schema_file_path_dir) as yaml_file:
             schema = yaml.safe_load(yaml_file)
        self.schema_config=(schema)
        
        
    def clean_df(self):

        df=pd.read_csv(self.data_cleaning_config.raw_data_file_path_dir)
        
        columns_dtypes_change_to_str = self.schema_config["cols_dtypes_change_to_str"]
        columns_to_drop=self.schema_config["cols_to_drop"]
        
        

        df["datesold"]=pd.to_datetime(df["datesold"], format="%m/%d/%Y",errors="coerce")
        df['bedrooms_str']=df['bedrooms'].replace([0,1,2,3,4,5],
                                                        ["Without_bedrooms", "One_bedrooms","Two_bedrooms","Three_bedrooms","Four_bedrooms","Five_bedrooms"])
        
        df[columns_dtypes_change_to_str]=df[columns_dtypes_change_to_str].astype("str")
        
        df["year"]=df["datesold"].dt.year.astype("str")

        df["month_name"]=df["datesold"].dt.month_name()

        df["day_name"]=df["datesold"].dt.day_name()

        df["Price_Per_Bedrooms"]=np.round(np.where(df['bedrooms'] != 0, df['price'] / df['bedrooms'], 0))

        df=df.drop(columns_to_drop,axis=1)

        df=df.rename(columns={'propertyType': 'Property_Type',
                              'postcode':'Region_Code'
                                })
        
        df.columns=df.columns.str.capitalize()

        df.to_csv(os.path.join(self.data_cleaning_config.cleaned_data_ingested_dir,CLEANED_DATA_FILE_DIR_NAME),index=False,header=True)

        
        #print(df.shape)    