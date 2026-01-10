import os 
import sys  

import numpy as np 
import dill 
import yaml 
from pandas import DataFrame

from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging 

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    
    except Exception as e:
        raise USvisaException(e,sys)


def write_yaml_file(file_path:str,
                    content:object,
                    replace:bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        # os.path.dirname extracts parent directory 
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,"w") as file:
            yaml.dump(content,file)
        
    except Exception as e:
        raise USvisaException(e,sys) from e # use original error and traceback
    
def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:
        with open(file_path,"rb") as f:
            obj = dill.load(f)
        
        logging.info("Exited load_object() in utils")
        return obj 
    
    except Exception as e:
        raise USvisaException(e,sys) from e 

def save_numpy_array_data(file_path: str,array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open("file_path","wb") as f:
            np.save(f,array)
    
    except Exception as e:
        raise USvisaException(e,sys) from e
    

def load_numpy_array_data(file_path: str):
    try:
        with open(file_path,"rb") as f:
            return np.load(f)
    
    except Exception as e:
        raise USvisaException(e,sys) from e

def save_object(file_path: str,obj: object):
    logging.info("Entered the save_object() in utils")
    
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as f:
            dill.dump(obj,f)
        
        logging.info("Exited save_object() of utils")
    
    except Exception as e:
        raise USvisaException(e,sys) from e

def drop_columns(df: DataFrame,cols: list) -> DataFrame:
    logging.info("Entered drop_columns() in utils")
    try:
        df = df.drop(columns=cols,axis=1)
        logging.info("Exited drop_columns() in utils")
        return df
    except Exception as e:
        raise USvisaException(e,sys) from e   