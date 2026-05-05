""
Audio Transcription API using OpenAI Whisper

This module provides a FastAPI-based REST API for converting audio files to text
using OpenAI's Whisper automatic speech recognition (ASR) model. The API supports
multiple audio formats and provides high-quality transcription with automatic
language detection capabilities.

Features:
    - High-quality transcription using Whisper models (tiny, base, small, medium, large)
    - Automatic language detection (99+ languages supported)
    - GPU acceleration via CUDA when available
    - Support for MP3, WAV, M4A, and FLAC audio formats
    - Automatic temporary file cleanup
    - Comprehensive error handling

Dependencies:
    - fastapi: Modern web framework for building APIs
    - uvicorn: ASGI server for running FastAPI applications
    - whisper: OpenAI's Whisper ASR model
    - torch: PyTorch for model inference

Usage:
    Run the application:
        $ python main.py
    
    The API will be available at http://0.0.0.0:8000
    Interactive documentation at http://0.0.0.0:8000/docs

    Transcribe an audio file:
        $ curl -X POST "http://localhost:8000/transcribe/" \
            -H "Content-Type: multipart/form-data" \
            -F "file=@audio.mp3"

Author: FastAPI/Whisper Integration
Version: 1.0.0
License: MIT
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
import torch

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Local Audio to Text API with Whisper",
    description="REST API for audio transcription using OpenAI's Whisper model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# Configuration
# =============================================================================

# Detect available compute device - prefer GPU for faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model for transcription
# Available models (in order of size/speed/accuracy):
#   - tiny: Fastest, lowest accuracy (~39 MB)
#   - base: Good balance (~74 MB)
#   - small: Better accuracy, moderate speed (~244 MB)
#   - medium: High accuracy, slower (~769 MB) [CURRENT DEFAULT]
#   - large: Best accuracy, slowest (~1550 MB)
#
# Note: Model loading may take several minutes on first run
#       as it downloads the model weights if not cached locally.
MODEL_NAME = "medium"
model = whisper.load_model(MODEL_NAME, device=device)
print(f"Loaded Whisper '{MODEL_NAME}' model on {device}")

# Supported audio file extensions for validation
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}

# =============================================================================
# API Endpoints
# =============================================================================

@app.post(
    "/transcribe/",
    response_description="Audio transcription result",
    status_code=200,
    summary="Transcribe audio file to text",
    tags=["Transcription"]
)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file to text using Whisper ASR model.
    
    This endpoint accepts audio files in MP3, WAV, M4A, or FLAC format
    and returns the transcribed text along with detected language.
    
    Parameters:
        file (UploadFile): The audio file to transcribe
            - Must be one of: .mp3, .wav, .m4a, .flac
            - File size limited by server configuration
            - Uploaded via multipart/form-data
    
    Returns:
        JSONResponse containing:
            - filename (str): Original filename of uploaded audio
            - transcription (str): Transcribed text from audio
            - language (str): Detected language of the audio
    
    Raises:
        HTTPException 400: If file format is not supported
        HTTPException 500: If transcription process fails
    
    Example:
        Request:
            POST /transcribe/
            Content-Type: multipart/form-data
            Body: file=@recording.mp3
        
        Response (200):
            {
                "filename": "recording.mp3",
                "transcription": "Hello world, this is a test.",
                "language": "English"
            }
    
    Notes:
        - The current implementation forces English transcription (language="en")
        - To enable automatic language detection, remove the language parameter
        - Temporary files are automatically cleaned up after processing
        - Processing time depends on audio length and model size
    """
    
    # ------------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------------
    
    # Extract and validate file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{file_extension}'. "
                f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        )
    
    # ------------------------------------------------------------------------
    # File Processing
    # ------------------------------------------------------------------------
    
    # Initialize temp file path for cleanup in error handler
    temp_file_path = ""
    
    try:
        # Save uploaded file to temporary location
        # Using NamedTemporaryFile with delete=False to allow Whisper to read it
        # The file will be manually deleted after transcription
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
            prefix="audio_"
        ) as temp_file:
            # Read uploaded file content asynchronously and write to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # --------------------------------------------------------------------
        # Transcription
        # --------------------------------------------------------------------
        
        # Perform speech-to-text transcription using Whisper model
        # language="en" forces English transcription
        # Remove this parameter to enable automatic language detection
        # (Whisper supports 99+ languages)
        result = model.transcribe(
            temp_file_path,
            language="en",  # Force English; remove for auto-detect
            fp16=device == "cuda"  # Use FP16 on GPU for faster inference
        )
        
        # --------------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------------
        
        # Remove temporary file after successful transcription
        # This is done before returning response to ensure cleanup even if
        # JSON serialization fails
        os.unlink(temp_file_path)
        temp_file_path = ""  # Mark as cleaned up
        
        # --------------------------------------------------------------------
        # Response
        # --------------------------------------------------------------------
        
        # Return transcription results as JSON response
        return JSONResponse(
            content={
                "filename": file.filename,
                "transcription": result["text"].strip(),
                "language": result["language"],
                "language_probability": result.get("language_probability", 1.0),
                "model": MODEL_NAME,
                "device": device
            },
            status_code=200
        )
    
    # ------------------------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------------------------
    
    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    
    except Exception as e:
        # Clean up temporary file if it exists and transcription failed
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                # Best effort cleanup - ignore errors during cleanup
                pass
        
        # Log error for debugging (in production, use proper logging)
        print(f"Transcription error: {type(e).__name__}: {str(e)}")
        
        # Raise HTTP 500 with error details
        # Note: In production, consider hiding internal error details
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


@app.get(
    "/",
    summary="Health check endpoint",
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        JSON response with API status and configuration details
    """
    return {
        "status": "healthy",
        "service": "Audio Transcription API",
        "model": MODEL_NAME,
        "device": device,
        "supported_formats": list(ALLOWED_EXTENSIONS)
    }


@app.get(
    "/health",
    summary="Health check endpoint (alternate)",
    tags=["Health"]
)
async def health_check_alt():
    """
    Alternate health check endpoint.
    
    Returns:
        JSON response with API status
    """
    return await health_check()


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    # Import uvicorn here to avoid unnecessary dependency for module imports
    import uvicorn
    
    # Configuration for Uvicorn ASGI server
    # host="0.0.0.0" makes the server accessible from external machines
    # port=8000 is the standard port for development APIs
    # Set debug=True for development (enables auto-reload on code changes)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )