import sys 

import numpy as np 
import pandas as pd 

from imblearn.combine import SMOTEENN

from sklearn.preprocessing import (StandardScaler,
                                   OrdinalEncoder,
                                   OneHotEncoder,
                                   PowerTransformer)
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 

from us_visa.constants.constant import TARGET_COLUMN,SCHEMA_FILE_PATH,CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import (DataIngestionArtifact,
                                            DataTransformationArtifact,
                                            DataValidationArtifact)
from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging 
from us_visa.utils.main_utils import (save_object,
                                      save_numpy_array_data,
                                      read_yaml_file,
                                      drop_columns)
from us_visa.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise USvisaException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise USvisaException(e,sys)
    
    def get_data_transformer_object(self) -> Pipeline:
        '''
        This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        '''

        try:
            logging.info("Got the numerical columns from schema_config file")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(drop='first',sparse_output=False)
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized std scaler,one hot encoder and ord encoder objects")

            oh_columns = self._schema_config["oh_columns"]
            or_columns = self._schema_config["or_columns"]
            transform_columns = self._schema_config["transform_columns"]
            num_features = self._schema_config["num_features"] 

            logging.info("Initializing PowerTransformer")

            transform_pipe = Pipeline([
                ("transformer",PowerTransformer(method="yeo-johnson"))
            ])

            preprocessor = ColumnTransformer([
                ("OneHotEncoder",oh_transformer,oh_columns),
                ("OrdinalEncoder",ordinal_encoder,or_columns),
                ("Transformer",transform_pipe,transform_columns),
                ("StandardScaler",numeric_transformer,num_features)
            ])

            logging.info("Created preprocessing object")
            return preprocessor

        except Exception as e:
            raise USvisaException(e,sys) 
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:

        '''
        Initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        '''
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()

                logging.info("Created the preprocessor pipeline object for data transformation")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Created the train features and test features of training dataset")
                
                input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
                logging.info("Created new feature called company_age")

                drop_cols = self._schema_config["drop_columns"]
                logging.info(f"Dropping columns {drop_cols} from the train dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df,cols=drop_cols)

                mapping = TargetValueMapping()._asdict()

                target_feature_train_df = (
                    target_feature_train_df
                    .map(mapping)
                    .astype(int)
                )

                input_feature_test_df =  test_df.drop(columns=[TARGET_COLUMN],axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
                logging.info("Created new feature company_age in test dataset")

                input_feature_test_df = drop_columns(df=input_feature_test_df,cols=drop_cols)
                logging.info(f"Dropping columns {drop_cols} from the test dataset")

                target_feature_test_df = (
                    target_feature_test_df
                    .map(mapping)
                    .astype(int)
                )

                logging.info("Built training features and testing features")

                logging.info("Applying the preprocessing pipeline to training dataframe and testing dataframe")
                
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr =  preprocessor.transform(input_feature_test_df)
                logging.info("Transformed the training and testing dataset")

                logging.info("Applying SMOTEENN on training and testing dataset to handle the class imbalance")
                smt = SMOTEENN(sampling_strategy='minority')

                input_feature_train_final,target_feature_train_final = smt.fit_resample(input_feature_train_arr,
                                                                                        target_feature_train_df)
                
                input_feature_test_final,target_feature_test_final = smt.fit_resample(input_feature_test_arr,
                                                                                      target_feature_test_df)
                logging.info("Creating train array and test array")

                train_arr = np.c_[input_feature_train_final,
                                  np.array(target_feature_train_final)]
                
                test_arr = np.c_[input_feature_test_final,
                                 np.array(target_feature_test_final)]
                
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved preprocessor object")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )


                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USvisaException(e,sys) 
        