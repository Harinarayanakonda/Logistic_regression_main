# src/interfaces/api/main.py
"""FastAPI application for the preprocessing pipeline."""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
import joblib
from src.core.preprocessing.pipeline_orchestrator import PreprocessingPipeline
from src.core.ml.inference import ModelInferencer
from .schemas import PreprocessingRequest, PreprocessingResponse
from src.config.paths import paths

app = FastAPI(
    title="Data Preprocessing Pipeline API",
    description="API for data preprocessing and model inference",
    version="1.0.0"
)

@app.post("/preprocess", response_model=PreprocessingResponse)
async def preprocess_data(file: UploadFile = File(...)):
    """Endpoint for preprocessing uploaded data."""
    try:
        # Load data
        file_format = Path(file.filename).suffix
        temp_path = paths.RAW_DATA_DIR / f"temp_upload{file_format}"
        
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
        
        loader = DataLoader()
        raw_df = loader.load_data(temp_path)
        
        # Run preprocessing pipeline
        pipeline = PreprocessingPipeline()
        processed_df = pipeline.run_pipeline(raw_df)
        artifact_file = pipeline.save_artifacts()
        
        # Prepare response
        response = PreprocessingResponse(
            message="Preprocessing completed successfully",
            original_shape=raw_df.shape,
            processed_shape=processed_df.shape,
            artifact_file=str(artifact_file),
            steps=pipeline.pipeline_steps
        )
        
        return response
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error during preprocessing: {str(e)}"}
        )

@app.post("/predict")
async def predict_data(file: UploadFile = File(...), model_file: str = None):
    """Endpoint for making predictions on new data."""
    try:
        # Load data
        file_format = Path(file.filename).suffix
        temp_path = paths.RAW_DATA_DIR / f"temp_upload{file_format}"
        
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
        
        loader = DataLoader()
        new_data = loader.load_data(temp_path)
        
        # Load model and artifacts
        if model_file is None:
            model_files = list(paths.MODELS_DIR.glob("model_*.pkl"))
            if not model_files:
                return JSONResponse(
                    status_code=404,
                    content={"message": "No trained models found"}
                )
            model_file = model_files[0]
        
        artifact_files = list(paths.MODELS_DIR.glob("model_artifacts_*.pkl"))
        if not artifact_files:
            return JSONResponse(
                status_code=404,
                content={"message": "No preprocessing artifacts found"}
            )
        artifact_file = artifact_files[0]
        
        # Make predictions
        inferencer = ModelInferencer()
        inferencer.load_model(model_file)
        inferencer.load_artifacts(artifact_file)
        
        predictions = inferencer.predict(new_data)
        
        return {
            "predictions": predictions.tolist(),
            "model_used": str(model_file),
            "artifacts_used": str(artifact_file)
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error during prediction: {str(e)}"}
        )