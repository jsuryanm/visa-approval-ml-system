import sys 
from pandas import DataFrame 

from us_visa.cloud_storage.aws_storage import SimpleStorageService
from us_visa.exception.exceptions import USvisaException
from us_visa.entity.estimator import UsVisaModel

class USvisaEstimator:
    '''
    This class is used to save and retrieve the us_visa model in s3 bucket and make prediction
    '''

    def __init__(self,bucket_name,model_path):
        '''
        Docstring for __init__

        :param bucket_name: Name of the model bucket
        :param model_path: Location of the model in bucket
        '''
        self.bucket_name = bucket_name 
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: UsVisaModel=None

    def is_model_present(self,model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name,
                                                 s3_key=model_path)
        except USvisaException as e:
            print(str(e))
            return False
    
    def load_model(self) -> UsVisaModel:
        '''
        Loads the model from model_path
        '''
        return self.s3.load_model(self.model_path,
                                  bucket_name=self.bucket_name)

    def save_model(self,from_file,remove: bool=False) -> None:
        '''
       Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        '''
        try:
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove)
        except Exception as e:
            raise USvisaException(str(e),sys)
        
    def predict(self,dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            
            return self.loaded_model.predict(dataframe=dataframe)

        except Exception as e:
            raise USvisaException(str(e),sys) 

