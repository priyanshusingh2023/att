# Audio Transcription API

A production-ready FastAPI service that converts audio files to text using OpenAI's Whisper model. This API provides high-quality transcription with automatic language detection and GPU acceleration support.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py

# Test the API
curl -X POST "http://localhost:8000/transcribe/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3"
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🎯 Features

- **High-Quality Transcription**: Uses Whisper medium model (configurable)
- **Multi-Language Support**: Auto-detects 99+ languages
- **GPU Acceleration**: Automatic CUDA support when available
- **Multiple Formats**: MP3, WAV, M4A, FLAC
- **FastAPI**: Modern async framework with automatic docs
- **Production-Ready**: Error handling, temp file cleanup, validation

## 📋 Requirements

- Python 3.7+
- 8GB+ RAM (16GB+ recommended)
- CUDA-capable GPU (optional, recommended)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/att.git
cd att

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn whisper torch

# For GPU support (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚦 Usage

### Start Server

```bash
python main.py
```

Server runs on `http://0.0.0.0:8000`

### API Endpoints

#### POST /transcribe/

Transcribe an audio file to text.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (audio file)

**Supported Formats:**
- MP3
- WAV
- M4A
- FLAC

**Response:**
```json
{
  "filename": "audio.mp3",
  "transcription": "Hello world, this is a test.",
  "language": "English"
}
```

**Error Responses:**
- `400`: Unsupported file format
- `500`: Transcription failed

### Examples

#### Python
```python
import requests

with open('audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/transcribe/',
        files={'file': f}
    )
print(response.json())
```

#### Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('audio.mp3'));

axios.post('http://localhost:8000/transcribe/', form, {
  headers: form.getHeaders()
}).then(console.log);
```

## ⚙️ Configuration

### Model Selection

Edit `main.py` line 15:
```python
# Options: tiny, base, small, medium, large
model = whisper.load_model("medium", device=device)
```

### Language Detection

Edit `main.py` line 37:
```python
# Auto-detect (remove language parameter):
result = model.transcribe(temp_file_path)

# Force specific language:
result = model.transcribe(temp_file_path, language="en")
```

## 📊 Performance

| Model | CPU | GPU | Accuracy |
|-------|-----|-----|----------|
| tiny | ~5s/min | ~1s/min | Low |
| base | ~10s/min | ~2s/min | Medium |
| small | ~20s/min | ~3s/min | High |
| medium | ~30s/min | ~5s/min | Very High |
| large | ~60s/min | ~10s/min | Best |

## 🔒 Security

- File type validation
- Automatic temp file cleanup
- Error handling prevents info leakage
- Consider adding rate limiting for production
- Implement authentication for public APIs

## 🧪 Testing

```bash
# Run tests (add your test suite)
python -m pytest tests/
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 🌟 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)

## 📞 Support

For issues and questions, please open an issue on GitHub.

## 📈 Roadmap

- [ ] Batch processing support
- [ ] Webhook notifications
- [ ] Async processing queue
- [ ] Multiple output formats (SRT, VTT)
- [ ] Speaker diarization
- [ ] Custom vocabulary support