import sys 
from typing import Tuple 

import importlib

import numpy as np 
import pandas as pd 
from pandas import DataFrame

from sklearn.pipeline import Pipeline 
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score)

from sklearn.model_selection import GridSearchCV

from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging 
from us_visa.utils.main_utils import (load_numpy_array_data,
                                      read_yaml_file,
                                      load_object,
                                      save_object)

from us_visa.entity.config_entity import ModelTrainerConfig

from us_visa.entity.artifact_entity import (DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ClassificationMetricArtifact)

from us_visa.entity.estimator import UsVisaModel

class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self,
                                    train: np.array,
                                    test: np.array) -> Tuple[object,object]:
        '''
        This method performs GridSearchCV to find the best model
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        '''
        try:
            logging.info("Performing GridSearchCV")

            # loading the npy array
            x_train,y_train = train[:,:-1],train[:,-1]
            x_test,y_test = test[:,:-1],test[:,-1]

            config  = read_yaml_file(self.model_trainer_config.model_config_file_path)

            grid_params = config["grid_search"]["params"]
            model_blocks = config["model_selection"]

            best_model = None 
            best_metric_artifact = None 

            for _,model_config in model_blocks.items():
                # loading the GridSearch model class 
                module = importlib.import_module(model_config["module"])
                model_class = getattr(module,model_config["class"])

                # load the model with fixed params
                model = model_class(**model_config.get("params",{}))

                gs = GridSearchCV(estimator=model,
                                           param_grid=model_config["search_param_grid"],
                                           **grid_params)
                
                logging.info(f"Running GridSearchCV for {model_class.__name__}")
                gs.fit(x_train,y_train)

                y_pred = gs.best_estimator_.predict(x_test)
                acc = accuracy_score(y_test,y_pred)

                if acc > self.model_trainer_config.expected_accuracy:
                    best_score = acc 
                    best_model = gs.best_estimator_
                    best_metric_artifact = ClassificationMetricArtifact(
                        f1_score=f1_score(y_test,y_pred),
                        precision_score=precision_score(y_test,y_pred),
                        recall_score=recall_score(y_test,y_pred)
                    )
            
            return best_model,best_metric_artifact,best_score
        
        except Exception as e:
            raise USvisaException(str(e),sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        '''
        Initiates the model trainer steps
        Output      :   Returns model trainer artifact
        '''

        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model,best_metric_artifact,best_score = self.get_model_object_and_report(train=train_arr,
                                                                                          test=test_arr)
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with higher accuracy score than baseline")
                raise Exception("No best model found with higher accuracy score than baseline")
            
            usvisa_model = UsVisaModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path")
            
            save_object(self.model_trainer_config.trained_model_file_path,usvisa_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=best_metric_artifact
            )

            logging.info(f"Model trainer artifact:{model_trainer_artifact}")
            
            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(str(e),sys)