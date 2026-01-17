import sys 

import pandas as pd 
from sklearn.pipeline import Pipeline

from us_visa.exception.exceptions import USvisaException
from us_visa.logger.logger import logging 

class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 0
        self.Denied: int = 1 
    
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))