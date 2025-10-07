from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import os
import tempfile
import torch

app = FastAPI(title="Local Audio to Text API with Whisper")

# Detect device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model (use 'medium' for good accuracy; change to 'small' for faster or 'large' for best)
model = whisper.load_model("medium", device=device)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and get its transcription.
    Supported formats: MP3, WAV, M4A, FLAC.
    """
    # Validate file type
    allowed_extensions = {".mp3", ".wav", ".m4a", ".flac"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use MP3, WAV, M4A, or FLAC.")

    # Save to temporary file
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Transcribe the audio
        result = model.transcribe(temp_file_path, language="en")  # Change language or remove for auto-detect

        # Clean up
        os.unlink(temp_file_path)

        # Return transcription
        return JSONResponse(content={
            "filename": file.filename,
            "transcription": result["text"],
            "language": result["language"]
        })

    except Exception as e:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)