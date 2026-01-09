import os 
import logging 
from datetime import datetime
from from_root import from_root

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(),"logs")

log_dir = "logs"

os.makedirs(log_path,exist_ok=True)

LOG_FILEPATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILEPATH,
                    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

logs_path = os.path.join(from_root(),log_dir,LOG_FILE)

if __name__ == "__main__":
    logging.info("This is a test log message")