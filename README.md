# Local Audio to Text API with Whisper

A FastAPI-based REST API that provides audio-to-text transcription using OpenAI's Whisper model. This service allows you to upload audio files and receive accurate transcriptions with language detection.

## Features

- **High-Quality Transcription**: Uses OpenAI's Whisper medium model for accurate speech recognition
- **Automatic Language Detection**: Supports multiple languages with auto-detection capabilities
- **GPU Acceleration**: Automatically uses CUDA-enabled GPUs when available, falls back to CPU
- **RESTful API**: Simple HTTP POST endpoint for easy integration
- **Multiple Audio Formats**: Supports MP3, WAV, M4A, and FLAC formats
- **FastAPI**: Modern, fast web framework with automatic API documentation

## Requirements

- Python 3.7+
- CUDA-capable GPU (optional, recommended for faster processing)
- 8GB+ RAM (16GB+ recommended for large audio files)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd att
```

2. Install dependencies:
```bash
pip install fastapi uvicorn whisper torch
```

3. (Optional) If you have a CUDA-enabled GPU, ensure PyTorch is installed with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Starting the Server

Run the API server:
```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

### API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Transcription Endpoint

**POST** `/transcribe/`

Upload an audio file to receive its transcription.

#### Request

- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (required): Audio file to transcribe
    - Supported formats: MP3, WAV, M4A, FLAC
    - Maximum file size: Limited by server configuration

#### Response

```json
{
  "filename": "example.mp3",
  "transcription": "This is the transcribed text from the audio file.",
  "language": "English"
}
```

#### Error Responses

- **400 Bad Request**: Unsupported file format or missing file
- **500 Internal Server Error**: Transcription failed (includes error details)

### Example Usage

#### Using curl

```bash
curl -X POST "http://localhost:8000/transcribe/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3"
```

#### Using Python

```python
import requests

url = "http://localhost:8000/transcribe/"
files = {"file": open("audio.mp3", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

#### Using JavaScript (Node.js)

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('audio.mp3'));

axios.post('http://localhost:8000/transcribe/', form, {
  headers: form.getHeaders()
})
.then(response => console.log(response.data))
.catch(error => console.error(error));
```

## Configuration

### Model Selection

The current implementation uses the "medium" Whisper model. You can change this in `main.py` line 15:

```python
# Available models: tiny, base, small, medium, large
model = whisper.load_model("medium", device=device)
```

- **tiny**: Fastest, lowest accuracy
- **base**: Good balance of speed and accuracy
- **small**: Better accuracy, moderate speed
- **medium**: High accuracy (current default)
- **large**: Best accuracy, slowest

### Language Settings

The current implementation forces English transcription (line 37). To enable automatic language detection, modify:

```python
# Current (English only):
result = model.transcribe(temp_file_path, language="en")

# Auto-detect language:
result = model.transcribe(temp_file_path)
```

## Performance

- **CPU Processing**: ~10-30 seconds per minute of audio (depending on model size)
- **GPU Processing**: ~2-5 seconds per minute of audio (with CUDA-enabled GPU)
- **Memory Usage**: ~2-4GB for medium model

## Security Considerations

- File validation prevents non-audio file uploads
- Temporary files are automatically cleaned up after processing
- Error handling prevents information leakage
- Consider adding rate limiting for production use
- Implement authentication for public deployments

## Development

### Running Tests

Add your test suite here when available.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the transcription model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PyTorch](https://pytorch.org/) for the deep learning framework

## Troubleshooting

### Out of Memory Errors

- Use a smaller model (tiny, base, or small)
- Process shorter audio files
- Reduce batch size if implementing batch processing

### Slow Processing

- Enable GPU acceleration with CUDA
- Use a smaller model
- Check system resources

### CUDA Not Available

- Verify GPU drivers are installed
- Check CUDA toolkit installation
- Ensure PyTorch is installed with CUDA support