# visa-approval-ml-system
An End-to-End machine learning system for US visa approval prediction, using automated data pipelines, model training and evaluation, FastAPI-based inference.

## Project Overview

This repository implements a **production-style ML system** that covers the complete lifecycle of a machine learning model:

- Data ingestion from MongoDB
- Data validation and drift detection using Evidently.ai
- Feature engineering and preprocessing
- Model training with hyperparameter tuning
- Model evaluation against a production model
- Model registry using AWS S3
- Inference via FastAPI
- UI for predictions
- Containerized deployment on AWS EC2
- CI/CD using GitHub Actions

---

## Problem Statement

Given historical US visa application data, predict whether a visa application will be:

- **Approved**
- **Rejected**

Prediction is done using features such as:
- Applicants level of education
- Continent 
- Region of employment
- Salary
- Applicant work experience
- Unit of wage (Hourly, Weekly, Monthly and Yearly)
- Requires any job training.
- Number of employees in company
- Number of years company has been in operation

Using the KNN Classifier or Random Forest Classifier (the model that is used for inferencing is the one that has higher cross validation f1 score higher than the benchmarked score).  

---
##  System Architecture

```
MongoDB
  ↓
Data Ingestion
  ↓
Data Validation (Schema + Drift Detection)
  ↓
Data Transformation (Feature Engineering + Encoding)
  ↓
Model Training (GridSearchCV)
  ↓
Model Evaluation (Compare with Production Model)
  ↓
Model Pusher
  ↓
AWS S3 (Model Registry)
  ↓
FastAPI Inference Service
  ↓
UI (HTML / Streamlit)
```

---

## Project Directory Structure
```
visa-approval-ml-system/
├── flowcharts/ # contains flowchart diagrams for the workflow of the pipelines
│ ├── data_ingestion.png
│ ├── data_validation.png
│ ├── data_transformation.png
│ │── model_trainer.png
│ ├── model_evaluation.png
│ └── model_pusher.png
├── us_visa/
│ ├── components/ # Core pipeline stages
│ │ ├── data_ingestion.py
│ │ ├── data_validation.py
│ │ ├── data_transformation.py
│ │ ├── model_trainer.py
│ │ ├── model_evaluation.py
│ │ └── model_pusher.py
│ │
│ ├── pipeline/
│ │ ├── training_pipeline.py
│ │ └── prediction_pipeline.py
│ │
│ ├── entity/ # Config & artifact schemas
│ │ ├── config_entity.py
│ │ ├── artifact_entity.py
│ │ └── estimator.py
│ │
│ ├── cloud_storage/
│ │ ├── aws_connection.py
│ │ └── aws_storage.py
│ │
│ ├── constants/
│ │ └── constant.py
│ │
│ ├── utils/
│ │ └── main_utils.py
│ │
│ ├── logger/
│ │ └── logger.py
│ │
│ └── exception/
│ └── exceptions.py
│
├── templates/
│ └── usvisa.html # UI template
│
├── static/
│ └── css/styles.css
│ 
├── app.py # FastAPI entrypoint
├── Dockerfile
├── requirements.txt
└── README.md
```
---

### The workflow process behind the pipelines 
- The project follows a **configuration-driven, modular pipeline architecture** suitable for production ML systems.
- Each pipeline stage is **independent, testable, and reusable**, with clear separation of configuration, logic, and outputs.

### Constants Layer (`constants/constant.py`)
- Centralized definition of:
  - Pipeline names and artifact directories
  - File and folder naming conventions
  - Target column
  - Model performance thresholds
  - AWS and application-level configuration
- Eliminates hardcoding and ensures consistency across the system.

### Entity Layer (`entity/`)
- Acts as the **contract layer** between pipeline stages.

**Config Entities (`config_entity.py`)**
- Define configuration classes for each pipeline stage.
- Create required directories and file paths using constants.
- Ensure components receive only required configuration.

**Artifact Entities (`artifact_entity.py`)**
- Define output objects for each pipeline stage.
- Store paths, metrics, and status flags.
- Enable explicit and traceable data flow between pipeline stages.

### Component Layer (`components/`)
- Contains the core logic for each pipeline stage.
- Each component:
  - Performs a single responsibility
  - Accepts a config object as input
  - Produces an artifact object as output
- Components include:
  - Data Ingestion
  - Data Validation
  - Data Transformation
  - Model Training
  - Model Evaluation
  - Model Pusher

### Pipeline Orchestration (`pipeline/`)
- Responsible for executing and coordinating pipeline stages.

**Training Pipeline**
- Initializes all pipeline configurations.
- Executes components sequentially.
- Passes artifacts between stages.
- Runs the full ML lifecycle from ingestion to model deployment.

**Prediction Pipeline**
- Loads the trained model from S3.
- Applies preprocessing and generates predictions.
- Used by FastAPI endpoints and the UI.
- Fully decoupled from the training pipeline.

### Application Layer (`app.py`)
- Exposes APIs for training and prediction.
- Connects the UI to the prediction pipeline.
- Acts as the user-facing interface of the ML system.



### 1️⃣ Training Pipeline (`training_pipeline.py`)

Runs the **entire ML lifecycle**:

1. **Data Ingestion**
   - Fetches data from MongoDB
   - Stores raw data in feature store

2. **Data Validation**
   - Schema validation
   - Column checks
   - Data drift detection using Evidently

3. **Data Transformation**
   - Feature engineering (e.g. company age)
   - Encoding (OneHot, Ordinal)
   - Scaling & power transforms
   - Class imbalance handling (SMOTEENN)

4. **Model Training**
   - GridSearchCV over multiple models
   - Performance threshold enforcement

5. **Model Evaluation**
   - Compare new model vs existing production model
   - Accept only if performance improves

6. **Model Pusher**
   - Uploads accepted model to AWS S3 (model registry)

---

### 2️⃣ Prediction Pipeline (`prediction_pipeline.py`)

Used during inference:

1. Accepts user input
2. Converts input to DataFrame
3. Loads model from S3
4. Applies preprocessing + prediction
5. Returns prediction result

---

## 🚀 How to Run the Project

### 🔧 Prerequisites

- Python 3.12.12
- Docker
- AWS account
- MongoDB connection
- EC2 instance with IAM role

---

### ▶️ Local Development

#### 1️⃣ Clone the repository
```bash
git clone https://github.com/jsuryanm/visa-approval-ml-system.git
```

#### 2️⃣ Create virtual environment
```bash
conda create --name myenv python=3.12
conda activate myenv
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```


### 4️⃣ Setup Environment Variables in .env file
```bash
MONGODB_CONNECTION_URL="your_mongodb_url"
AWS_ACCESS_KEY_ID="AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ID="AWS_SECRET_ACCESS_KEY_ID"
```

---

### Setup for Cloud Deployment with EC-2 and GitHub Actions CI/CD

#### 1. Login to AWS console.

#### 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

  	3. AmazonS3FullAccess

	
#### 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.ap-southeast-1.amazonaws.com/visarepo

	
#### 4. Create EC2 machine (Ubuntu) 

#### 5. Open EC2 and Install docker in EC2 Machine:
	
	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
#### 6. Configure EC2 as self-hosted runner:
    setting -> actions -> runner -> new self hosted runner -> choose linux os -> then run command one by one


#### 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - MONGODB_CONNECTION_URL

---
### Future improvements to implement
- Scalable Inference with ALB
- Streamlit UI frontend deployment
- Model Tracking and Experiment Management with MLFlow








