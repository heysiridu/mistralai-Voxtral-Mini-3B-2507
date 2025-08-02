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
| Model           | `mistralai/Voxtral-Mini-3B-2507`     |
| Context Length  | 32k tokens (30–40 min audio)          |
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

## ⚙️ Setup: Run Locally (optional)

Before you test locally, make sure you have [uv Python package and project manager](https://docs.astral.sh/uv/) installed. This will save you time for adjusting version conflicts. Make sure to follow all the guidelines to have uv in your folder.

```
pip install uv
```

1. **Clone the repo and install dependencies**

```bash
git clone git@github.com:heysiridu/voxtral_model_deployment_v2.git
cd voxtral_model_deployment_v2
uv pip install -r requirements.txt
```

2. **Start the service**

```bash
bentoml serve service:VoxtralAudioService --reload
```

Visit [http://localhost:3000](http://localhost:3000) for Swagger UI.

Notes: You can skip this step and deploy to the cloud directly.

---



## ☁️ Deploy to BentoCloud

After testing locally, get the **API key** from **Hugging Face website**, deploy to [BentoCloud](https://cloud.bentoml.com/) for scalability:

1. **Login and build**
   Login will redirect you to the BentoML Cloud page to create a token. After successfully running **`bentoml build`**, you'll see large BentoML ASCII art displayed in your terminal.

```bash
bentoml cloud login
bentoml build
```

2. **Deploy**

```
bentoml deploy . -n your-deployment-name
```

3. **Get the endpoint**

    You'll receive a URL like **`https://voxtral.bentoml.app/v1`** 

4. **Test on the Cloud**
   Use the **`female.wav`** file to test your deployment.

   ```
   curl -X POST "https://voxtral-audio-service-v4-44dee3e6.mt-guc1.bentoml.ai/transcribe_audio" \
          -H "Content-Type: multipart/form-data" \
          -F "audio_file=@female.wav" \
          --max-time 600
   ```

---

## 📚 Resources

* 🔗 [Hugging Face Voxtral Model](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
* 📘 [BentoML Documentation](https://docs.bentoml.com)
* 🧠 [vLLM Inference Guide
  ](https://docs.vllm.ai)

---
