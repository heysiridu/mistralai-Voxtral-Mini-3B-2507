
# 🎙️ Self-Host Voxtral-Mini-3B-2507 with BentoML

Follow this guide to self-host the [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) audio-language model using BentoML. This service enables high-performance audio **transcription**, **translation**, and **audio Q\&A** with support for popular audio formats.


If your team doesn’t already have access to BentoCloud, use the buttons below to get started.

[![Deploy on BentoCloud](https://img.shields.io/badge/Deploy_on_BentoCloud-d0bfff?style=for-the-badge)](https://testversiona.cloud.bentoml.com/deployments?q=)
[![Talk to sales](https://img.shields.io/badge/Talk_to_sales-eefbe4?style=for-the-badge)](https://bentoml.com/contact)

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

---

## 📌 Model Overview

| Property        | Value                                  |
| --------------- | -------------------------------------- |
| Model           | `mistralai/Voxtral-Mini-3B-2507`       |
| Context Length  | 32k tokens (30–40 min audio)           |
| GPU VRAM Needed | \~9.5 GB                               |
| Audio Formats   | WAV, MP3, M4A, FLAC, AAC, OGG, WMA     |
| Capabilities    | Transcription, translation, audio Q\&A |

---

## 🧱 Prerequisites

* Access to the [Voxtral model on Hugging Face](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
* Python 3.11+
* CUDA 12.1+ GPU (≥10GB VRAM recommended)
* [BentoML 1.2.23+](https://pypi.org/project/bentoml/)

---

## ⚙️ Setup: Run Locally

1. **Clone the repo and install dependencies**

```bash
git clone https://github.com/your-org/voxtral-model-deployment.git
cd voxtral_model_deployment_v2
pip install -r requirements.txt
```

2. **Start the service**

```bash
bentoml serve service:VoxtralAudioService --reload
```

Visit [http://localhost:3000](http://localhost:3000) for Swagger UI.

---

## 🧪 API Endpoints

### 🔍 Health Check
Run the code:

```bash
GET /health
```

**Example response:**

```json
{
  "status": "healthy",
  "model": "mistralai/Voxtral-Mini-3B-2507",
  "gpu_available": "true",
  "gpu_count": "1"
}
```

---

### 📝 Audio Transcription
Run the code:

```bash
POST /transcribe_audio
```

**Example:**

```bash
curl -X POST http://localhost:3000/transcribe_audio \
  -F "audio=@test_audio_samples/short_test.wav" \
  -F 'data={"language": "en", "temperature": 0.1, "max_tokens": 1024}'
```

---

### ❓ Audio Q\&A
Run the code:
```bash
POST /audio_qa
```

**Example:**

```bash
curl -X POST http://localhost:3000/audio_qa \
  -F "audio=@test_audio_samples/medium_test.wav" \
  -F 'data={"question": "What is the topic?", "temperature": 0.7}'
```

---

## ☁️ Deploy to BentoCloud

After testing locally, get the API key from Hugging Face website, deploy to [BentoCloud](https://cloud.bentoml.com/) for scalability:

1. **Login and set secret**

```bash
bentoml cloud login
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN
```

2. **Deploy**

```bash
bentoml deploy service:VoxtralAudioService --secret huggingface
```

3. **Use the endpoint**

You'll receive a URL like `https://voxtral.bentoml.app/v1`

---

## 📚 Resources

* 🔗 [Hugging Face Voxtral Model](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
* 📘 [BentoML Documentation](https://docs.bentoml.com)
* 🧠 [vLLM Inference Guide](https://docs.vllm.ai)

---
