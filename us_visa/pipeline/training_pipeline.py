import os 
import sys 

from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging

from us_visa.components.data_ingestion import DataIngestion

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact 

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Retrieving the data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the training set and testing set from MongoDB")
            return data_ingestion_artifact
        
        except Exception as e:
            raise USvisaException(e,sys) from e
        
    def run_pipeline(self) -> None:
        '''
        This method is responsible for running the entire pipeline
        
        :param self: Description
        '''

        try:
            data_ingestion_artifact = self.start_data_ingestion()
        
        except Exception as e: 
            raise USvisaException(e,sys) from e