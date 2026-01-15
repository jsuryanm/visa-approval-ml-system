from us_visa.configuration.mongo_db_connection import MongoDBClient

if __name__ == "__main__":
    connection = MongoDBClient()
    print(connection)