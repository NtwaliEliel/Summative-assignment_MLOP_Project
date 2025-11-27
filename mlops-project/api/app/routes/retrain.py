"""Retrain route for the Iris Classification API."""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from app.services.retrain_service import get_retrain_service
from app.services.model_service import get_model_service
from app.utils import format_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrain", tags=["retraining"])


@router.post("", response_model=dict)
async def retrain(file: UploadFile = File(...)):
    """
    Retrain the model with a new CSV file containing Iris data.
    
    The CSV must contain columns: sepal_length, sepal_width, petal_length, petal_width, target
    where target is 0 (setosa), 1 (versicolor), or 2 (virginica).
    
    Args:
        file: Uploaded CSV file
    
    Returns:
        JSON response with retraining status and evaluation metrics
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail=format_response(
                    status="error",
                    message="File must be a CSV file",
                    data={"filename": file.filename}
                )
            )
        
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail=format_response(
                    status="error",
                    message="Uploaded file is empty",
                    data={}
                )
            )
        
        retrain_service = get_retrain_service()
        
        # Save uploaded file
        csv_path = retrain_service.save_uploaded_csv(content, file.filename)
        
        # Validate CSV structure
        is_valid, error_msg = retrain_service.validate_csv(csv_path)
        if not is_valid:
            # Clean up invalid file
            csv_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=format_response(
                    status="error",
                    message=f"Invalid CSV file: {error_msg}",
                    data={
                        "required_columns": [
                            "sepal_length", 
                            "sepal_width", 
                            "petal_length", 
                            "petal_width", 
                            "target"
                        ],
                        "target_values": [0, 1, 2],
                        "error": error_msg
                    }
                )
            )
        
        # Retrain model
        logger.info(f"Starting retrain with file: {file.filename}")
        metrics = retrain_service.retrain_from_csvs([csv_path])
        
        # Reload model in model service
        model_service = get_model_service()
        model_service.reload_model()
        
        logger.info(f"Retrain completed successfully. Accuracy: {metrics['accuracy']:.3f}")
        
        return format_response(
            status="success",
            message="Model retrained successfully",
            data={
                "metrics": {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"]
                },
                "best_params": metrics["best_params"],
                "classification_report": metrics["classification_report"],
                "confusion_matrix": metrics["confusion_matrix"],
                "samples": {
                    "train": metrics["train_samples"],
                    "test": metrics["test_samples"],
                    "total": metrics["total_samples"]
                }
            }
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error in retrain: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=format_response(
                status="error",
                message="Data validation failed",
                data={"error": str(e)}
            )
        )
    except Exception as e:
        logger.error(f"Unexpected error in retrain: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=format_response(
                status="error",
                message="An unexpected error occurred during retraining",
                data={"error": str(e)}
            )
        )

