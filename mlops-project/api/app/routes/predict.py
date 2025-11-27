"""Prediction route for the Iris Classification API."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from app.services.model_service import get_model_service
from app.utils import format_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    sepal_length: float = Field(..., ge=0.0, le=20.0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0.0, le=20.0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0.0, le=20.0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0.0, le=20.0, description="Petal width in cm")
    
    @validator('*', pre=True)
    def validate_numeric(cls, v):
        """Ensure all values are numeric."""
        if not isinstance(v, (int, float)):
            try:
                return float(v)
            except (ValueError, TypeError):
                raise ValueError(f"Value must be numeric, got {type(v)}")
        return float(v)


@router.post("", response_model=dict)
async def predict(request: PredictRequest):
    """
    Predict the Iris species from sepal and petal measurements.
    
    Args:
        request: PredictRequest with sepal_length, sepal_width, petal_length, petal_width
    
    Returns:
        JSON response with label, label_id, and probability
    """
    try:
        model_service = get_model_service()
        
        result = model_service.predict_single(
            sepal_length=request.sepal_length,
            sepal_width=request.sepal_width,
            petal_length=request.petal_length,
            petal_width=request.petal_width
        )
        
        logger.info(f"Prediction made: {result['label']} (probability: {result['probability']:.3f})")
        
        return format_response(
            status="success",
            message="Prediction completed successfully",
            data=result
        )
        
    except RuntimeError as e:
        logger.error(f"Model service error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=format_response(
                status="error",
                message="Model not available. Please ensure model files are present.",
                data={"error": str(e)}
            )
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=format_response(
                status="error",
                message="An unexpected error occurred during prediction",
                data={"error": str(e)}
            )
        )

