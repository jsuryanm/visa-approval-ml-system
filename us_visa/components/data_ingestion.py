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
            visa_df = usvisa_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of visa_df loaded from mongodb:{visa_df.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Saving the exported data into feature store file path:{feature_store_file_path}")
            visa_df.to_csv(feature_store_file_path,index=False,header=True)
            return visa_df
              
        except Exception as e:
            raise USvisaException(e,sys)
        
    def split_data_as_train_test(self,visa_df: DataFrame) -> None:
        '''
        This method splits the visa_df into train_set and test_set based based on the split ratio
        
        Output: Folder is created in s3 bucket
        On Failure: Write an exception log and then raise the exception
        '''
        
        try:
            train_set,test_set = train_test_split(visa_df,
                                                  test_size=self.data_ingestion_config.train_test_split_ratio,
                                                  random_state=42)
            
            logging.info("Performed train_test_split on visa_df")
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info("Exporting train and test file path.")
            
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            
            logging.info("Exported train and test file path")     

        except Exception as e:
            raise USvisaException(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        '''
        This method initiates data ingestion components of the training pipeline 
        
        Output: train set and test set are returned as the artifacts of data ingestion components
        On Failuire: Write an exception log and raise an exception
        :rtype: DataIngestionArtifact
        '''

        try:
            visa_df = self.export_data_into_feature_store()
            logging.info("Retrieved the data from MongoDB")

            self.split_data_as_train_test(visa_df)
            logging.info("Performed train test split on dataset")

            data_ingestion_artifact  = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                             test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact:{data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e,sys)
