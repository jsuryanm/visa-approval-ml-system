import sys 

from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging 
from us_visa.entity.artifact_entity import ModelPusherArtifact,ModelEvaluationArtifact
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.s3_estimator import USvisaEstimator

class ModelPusher: 
    def __init__(self,
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                                model_path=model_pusher_config.s3_model_key_path)
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        '''
        Initiates all steps of model pusher
        '''
        try:
            logging.info("Uploading artifacts into s3 bucket")

            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)
            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact:{model_pusher_artifact}")

            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(str(e),sys)