import os 
import sys 

from pandas import DataFrame 
from sklearn.model_selection import train_test_split

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging 
from us_visa.data_access.us_visa_data import UsVisaData

class DataIngestion:
    def __init__(self,
                 data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        
        except Exception as e:
            raise USvisaException(e,sys)
        
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            usvisa_data = UsVisaData()
            visa_df = usvisa_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config)
            logging.info(f"Shape of visa_df loaded from mongodb:{visa_df.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            
              
        except Exception as e:
            raise USvisaException(e,sys)