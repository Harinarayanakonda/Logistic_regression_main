# src/interfaces/api/schemas.py
"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel
from typing import List

class PreprocessingRequest(BaseModel):
    """Request schema for preprocessing endpoint."""
    file_format: str
    target_column: str = None

class PreprocessingResponse(BaseModel):
    """Response schema for preprocessing endpoint."""
    message: str
    original_shape: tuple
    processed_shape: tuple
    artifact_file: str
    steps: List[str]