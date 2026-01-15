from us_visa.data_access.us_visa_data import UsVisaData
from us_visa.constants.constant import COLLECTION_NAME,DATABASE_NAME

if __name__ == "__main__":
    # connection = MongoDBClient()
    # print(connection)
    visa_data_obj = UsVisaData()
    visa_df = visa_data_obj.export_collection_as_dataframe(COLLECTION_NAME)
    print(visa_df) 