# Voxtral-Mini-3B-2507 BentoML Deployment

Production-ready BentoML service for the Mistral Voxtral-Mini-3B-2507 audio-language model, optimized for transcription, translation, and audio Q&A tasks.

## Model Overview

- **Model**: `mistralai/Voxtral-Mini-3B-2507`
- **Capabilities**: Audio transcription, translation, understanding, and Q&A
- **Context Length**: 32k tokens (30-40 minutes of audio)
- **Memory Requirement**: ~9.5GB GPU RAM
- **Supported Audio Formats**: WAV, MP3, M4A, FLAC, AAC, OGG, WMA

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.1+ compatible GPU (minimum 10GB VRAM recommended)
- BentoML 1.2.23+
- Docker (for containerized deployment)

### Local Development Setup

1. **Clone and Navigate**
   ```bash
   cd voxtral_model_deployment_v2
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables** (Optional)
   ```bash
   export VOXTRAL_MODEL_NAME="mistralai/Voxtral-Mini-3B-2507"
   export VOXTRAL_GPU_MEMORY_UTIL=0.85
   export VOXTRAL_MAX_BATCH_SIZE=4
   ```

4. **Start the Service**
   ```bash
   bentoml serve service:VoxtralAudioService --reload
   ```

5. **Test the Service**
   ```bash
   curl -X GET http://localhost:3000/health
   ```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "mistralai/Voxtral-Mini-3B-2507",
  "gpu_available": "true",
  "gpu_count": "1"
}
```

### 2. Audio Transcription
```bash
POST /transcribe_audio
```

**Request (multipart/form-data):**
- `audio`: Audio file (WAV, MP3, M4A, etc.)
- `data`: JSON with parameters

**Example:**
```bash
curl -X POST http://localhost:3000/transcribe_audio \
  -F "audio=@test_audio_samples/short_test_5s.wav" \
  -F 'data={"language": "en", "temperature": 0.1, "max_tokens": 1024}'
```

**Response:**
```json
{
  "text": "Hello, this is a test of the Voxtral audio transcription service.",
  "duration_seconds": 5.2,
  "model_info": {
    "model_name": "mistralai/Voxtral-Mini-3B-2507",
    "context_length": "32768"
  },
  "processing_time": 2.34
}
```

### 3. Audio Q&A
```bash
POST /audio_qa
```

**Request (multipart/form-data):**
- `audio`: Audio file
- `data`: JSON with question and parameters

**Example:**
```bash
curl -X POST http://localhost:3000/audio_qa \
  -F "audio=@test_audio_samples/medium_test_30s.wav" \
  -F 'data={"question": "What is the main topic discussed?", "temperature": 0.7, "max_tokens": 2048}'
```

**Response:**
```json
{
  "text": "The main topic discussed in the audio is...",
  "duration_seconds": 30.1,
  "model_info": {
    "model_name": "mistralai/Voxtral-Mini-3B-2507",
    "context_length": "32768"
  },
  "processing_time": 4.67
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXTRAL_MODEL_NAME` | `mistralai/Voxtral-Mini-3B-2507` | Model identifier |
| `VOXTRAL_GPU_MEMORY_UTIL` | `0.85` | GPU memory utilization (0.0-1.0) |
| `VOXTRAL_MAX_BATCH_SIZE` | `4` | Maximum batch size |
| `VOXTRAL_MAX_MODEL_LEN` | `32768` | Maximum context length |
| `VOXTRAL_TEMPERATURE` | `0.7` | Default sampling temperature |
| `VOXTRAL_REQUEST_TIMEOUT` | `300` | Request timeout in seconds |

### Audio Processing Settings

- **Sample Rate**: 16kHz (automatically resampled)
- **Max Duration**: 40 minutes (2400 seconds)  
- **Supported Formats**: WAV, MP3, M4A, FLAC, AAC, OGG, WMA
- **Max File Size**: Limited by available memory

## Deployment

### BentoML Cloud Deployment

1. **Build the Bento**
   ```bash
   bentoml build
   ```

2. **Deploy to BentoML Cloud**
   ```bash
   bentoml deploy <bento_tag> --cluster-name <cluster>
   ```

### Docker Deployment

1. **Build Docker Image**
   ```bash
   bentoml containerize voxtral-mini-3b-audio:latest -t voxtral-audio:latest
   ```

2. **Run Container**
   ```bash
   docker run --gpus all -p 3000:3000 voxtral-audio:latest
   ```

### Kubernetes Deployment

1. **Generate Kubernetes Manifests**
   ```bash
   bentoml generate kubernetes voxtral-mini-3b-audio:latest
   ```

2. **Apply to Cluster**
   ```bash
   kubectl apply -f bentoml-deployment.yaml
   ```

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Load Testing
```bash
# Install artillery
npm install -g artillery

# Run load test
artillery run load-test.yml
```

### Manual Testing

1. **Test Transcription**
   ```python
   import requests
   
   with open("test_audio_samples/short_test_5s.wav", "rb") as f:
       response = requests.post(
           "http://localhost:3000/transcribe_audio",
           files={"audio": f},
           data={"data": '{"temperature": 0.1}'}
       )
   print(response.json())
   ```

2. **Test Q&A**
   ```python
   import requests
   
   with open("test_audio_samples/medium_test_30s.wav", "rb") as f:
       response = requests.post(
           "http://localhost:3000/audio_qa",
           files={"audio": f},
           data={"data": '{"question": "Summarize the content", "temperature": 0.7}'}
       )
   print(response.json())
   ```

## Performance Optimization

### GPU Memory Management
- Use `gpu_memory_utilization=0.85` for optimal performance
- Monitor GPU memory usage with `nvidia-smi`
- Adjust `max_batch_size` based on available VRAM

### Scaling Configuration
- **Min Replicas**: 1
- **Max Replicas**: 3
- **Auto-scaling**: Based on CPU (70%) and memory (80%) utilization
- **Concurrency**: Limited to 2 concurrent requests per instance

### Audio Processing Optimization
- Audio is automatically resampled to 16kHz
- Large files are processed in chunks
- Memory is released immediately after processing

## Monitoring and Logging

### Health Monitoring
```bash
# Check service health
curl http://localhost:3000/health

# Monitor GPU usage
nvidia-smi -l 1
```

### Logging Configuration
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Metrics Collection
- Request latency tracking
- GPU memory usage monitoring
- Error rate tracking
- Audio processing duration metrics

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   - Reduce `gpu_memory_utilization` to 0.7
   - Decrease `max_batch_size` to 2
   - Process shorter audio segments

2. **Model Loading Timeout**
   - Increase `MODEL_LOAD_TIMEOUT` to 900 seconds
   - Ensure sufficient disk space for model cache
   - Check internet connectivity for model download

3. **Audio Processing Errors**
   - Verify audio format is supported
   - Check file size limits (< 100MB recommended)
   - Ensure audio is not corrupted

4. **Slow Response Times**
   - Use GPU-optimized Docker images
   - Enable tensor parallelism for larger models
   - Optimize audio preprocessing pipeline

### Debug Mode
```bash
BENTOML_LOG_LEVEL=DEBUG bentoml serve service:VoxtralAudioService
```

### Resource Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check Docker container stats
docker stats voxtral-audio
```

## Security Considerations

- Input validation for all audio files
- Rate limiting to prevent abuse
- Secure file handling with automatic cleanup
- Non-root container execution
- Network security policies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

- **Issues**: Create GitHub issues for bugs and feature requests
- **Documentation**: Check the official BentoML documentation
- **Community**: Join the BentoML Discord community

## Changelog

### v2.0.0
- Initial production-ready deployment
- vLLM integration with audio support
- Multi-format audio processing
- Auto-scaling configuration
- Comprehensive monitoring setup