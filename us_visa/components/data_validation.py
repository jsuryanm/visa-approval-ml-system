import os 
import sys 
import json 

import pandas as pd 
from pandas import DataFrame

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report 
from evidently.presets import DataDriftPreset,DataSummaryPreset

from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging 
from us_visa.utils.main_utils import  read_yaml_file,write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants.constant import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        '''
        
        :param data_ingestion_artifact: Output of data ingestion stage that creates train.csv,test.csv 
        :param data_validation_config: configuration for data validation
        '''

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise USvisaException(str(e),sys)
    
    def validate_number_of_columns(self,visa_df: DataFrame):
        """
        This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(visa_df.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column:[{status}]")
            return status
        
        except Exception as e:
            raise USvisaException(str(e),sys)
    
    def is_column_exist(self,visa_df: DataFrame) -> bool:
        """
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = visa_df.columns 
            missing_numerical_columns = []
            missing_categorical_columns = []

            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column:{missing_numerical_columns}")
            
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            
            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical column:{missing_categorical_columns}")
            
            return False if len(missing_categorical_columns) > 0 or len(missing_numerical_columns) > 0 else True
        
        except Exception as e:
            raise USvisaException(str(e),sys)
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e,sys)
    
    def detect_dataset_drift(self,reference_df: DataFrame,current_df: DataFrame) -> bool:
        """
        This method validates if drift is detected
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            data_definition = DataDefinition(
                numerical_columns=self._schema_config['numerical_columns'],
                categorical_columns=self._schema_config['categorical_columns']
            ) 

            reference_dataset = Dataset.from_pandas(
                data=reference_df,
                data_definition=data_definition
            )

            current_dataset = Dataset.from_pandas(
                data=current_df,
                data_definition=data_definition
            )

            report = Report(metrics=[
                DataDriftPreset()
            ])

            run = report.run(current_data=current_dataset,
                       reference_data=reference_dataset)
            
            run_dict = run.dict()
            
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path,
                            content=run_dict)
            
            metrics = run_dict["metrics"]

            drifted_columns_metric = next(
                m for m in metrics 
                if m['metric_name'].startswith("DriftedColumnsCount")
            )

            drifted_share = drifted_columns_metric["value"]["share"]
            drifted_threshold = drifted_columns_metric["config"]["drift_share"]
            drifted_count  = drifted_columns_metric["value"]["count"]

            drift_status = drifted_share >= drifted_threshold

            # n_features = drift_result["number_of_columns"]
            # n_drifted_features = drift_result["number_of_drifted_columns"]
            # drift_status = drift_result["dataset_drift"]

            # logging.info(f"{n_drifted_features}/{n_features} features drifed")
            # logging.info(f"Dataset drift detected:{drift_status}")

            logging.info(f"{int(drifted_count)} columns drifted"
                         f"({drifted_share:.2%}), threshold={drifted_threshold:.2%}")
            
            logging.info(f"Dataset drift detected:{drift_status}")

            return drift_status
        
        except Exception as e:
            raise USvisaException(str(e),sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df,test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status = self.validate_number_of_columns(visa_df=train_df)
            logging.info(f"All required columns present in training dataframe:{status}")

            if not status:
                validation_error_msg += f"Columns are missing training dataframe"
            status = self.validate_number_of_columns(visa_df=test_df)
            
            logging.info(f"All columns present in testing dataframe:{status}")
            if not status:
                validation_error_msg += f"Columns are missing in the testing dadtaframe"
            
            status = self.is_column_exist(visa_df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe"
            status = self.is_column_exist(visa_df=test_df)

            if not status:
                validation_error_msg += f"columns are missing in a dataframe"
            
            validation_status = len(validation_error_msg) == 0
            
            if validation_status:
                drift_status  = self.detect_dataset_drift(train_df,test_df)
                if drift_status:
                    logging.info("Drift detected")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error:{validation_error_msg}")
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact:{data_validation_artifact}")
            
            return data_validation_artifact 
        except Exception as e:
            raise USvisaException(str(e),sys)