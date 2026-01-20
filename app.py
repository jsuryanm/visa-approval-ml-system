from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware 
# enables cross origin sharing when frontend is hosted on a different port
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from us_visa.pipeline.prediction_pipeline import USvisaData,USvisaClassifier
from us_visa.pipeline.training_pipeline import TrainingPipeline
from uvicorn import run as app_run 
from us_visa.constants.constant import APP_HOST,APP_PORT
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS setting this allows request from any frontend
origins = ["*"] 


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class VisaPredictionRequest(BaseModel):
    continent: str
    education_of_employee: str 
    has_job_experience: str
    requires_job_training: str 
    no_of_employees: int
    region_of_employment: str 
    prevailing_wage: float 
    unit_of_wage: str
    full_time_position: str
    company_age: int

@app.post("/predict")

async def predict_visa_status(request: VisaPredictionRequest):
    try:
        usvisa_data = USvisaData(
            continent=request.continent,
            education_of_employee=request.education_of_employee,
            has_job_experience=request.has_job_experience,
            requires_job_training=request.requires_job_training,
            no_of_employees=request.no_of_employees,
            region_of_employment=request.region_of_employment,
            prevailing_wage=request.prevailing_wage,
            unit_of_wage=request.unit_of_wage,
            full_time_position=request.full_time_position,
            company_age=request.company_age
            )
        
        usvisa_df = usvisa_data.get_usvisa_input_data_frame()
        
        model = USvisaClassifier()
        prediction = model.predict(dataframe=usvisa_df)[0]
        
        visa_status = "Certified" if prediction == 1 else "Denied"
        
        return JSONResponse(status_code=200,
                            content={"status":True,
                                     "prediction":visa_status})
    
    except Exception as e:
        raise JSONResponse(status_code=500,
                           content={"status":False,"error":str(e)})
    

@app.get("/train")
async def train_model():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return JSONResponse(status_code=200,
                            content={"status":True,
                                     "message":"Training Successful"})
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"status":False,
                                     "error":str(e)}) 

@app.get("/")
async def root():
    return JSONResponse(status_code=200,
                        content={"message":"Welcome to the US Visa Prediction API"})

if __name__ == "__main__":
    app_run(app,host=APP_HOST,port=APP_PORT)