import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union

import bentoml

# Use bentoml.importing() context for runtime-only dependencies
with bentoml.importing():    
    import torch.nn.functional as F
    from vllm import AsyncLLMEngine, EngineArgs
    from vllm.sampling_params import SamplingParams

import librosa
import numpy as np
import torch
from pydantic import BaseModel, Field

from config import VoxtralConfig

ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger = logging.getLogger("bentoml")
logger.addHandler(ch)
logger.setLevel(logging.INFO)

class AudioResponse(BaseModel):
    text: str
    duration_seconds: float
    model_info: Dict[str, str]
    processing_time: float

@bentoml.service(
    name="voxtral-audio-service",
    resources={"gpu": 1, "gpu_type": "nvidia-tesla-t4", "memory": "16Gi"},
    traffic={"timeout": 300},
)
class VoxtralAudioService:
    def __init__(self):
        self.config = VoxtralConfig()
        self.engine = None
        # Note: vLLM engine will be initialized on first request
    
    async def _load_model(self):
        """Initialize vLLM AsyncLLMEngine with audio support"""
        try:
            logger.info(f"Loading model: {self.config.MODEL_NAME}")
            
            # Configure vLLM engine arguments
            engine_args = EngineArgs(
                model=self.config.MODEL_NAME,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                gpu_memory_utilization=0.8,
                max_model_len=self.config.MAX_MODEL_LEN,
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            logger.info("vLLM engine loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _process_audio_file(self, audio_file: Path, target_sr: int = 16000) -> np.ndarray:
        """Process uploaded audio file to numpy array"""
        try:
            # Load and resample audio directly from Path
            audio_array, sr = librosa.load(str(audio_file), sr=target_sr, mono=True)
            logger.info(f"Processed audio: {len(audio_array)} samples at {sr}Hz")
            return audio_array
                
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise ValueError(f"Failed to process audio file: {str(e)}")
    
    async def _generate_response(self, text_prompt: str, audio_file_path: Path, temperature: float, max_tokens: int) -> str:
        """Generate response using vLLM with audio support"""
        try:
            # Create sampling parameters for vLLM
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.config.TOP_P,
                repetition_penalty=self.config.REPETITION_PENALTY,
            )
            
            # vLLM format: prompt with multi_modal_data
            request_data = {
                "prompt": text_prompt,
                "multi_modal_data": {
                    "audio": str(audio_file_path)
                }
            }
            
            # Generate using vLLM async engine
            results = await self.engine.generate(
                **request_data,
                sampling_params=sampling_params
            )
            
            # Extract generated text
            generated_text = results[0].outputs[0].text
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    @bentoml.api
    async def transcribe_audio(
        self,
        audio_file: Path,
        language: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> AudioResponse:
        """Transcribe audio to text"""
        # Initialize engine if not already done
        if self.engine is None:
            await self._load_model()
            
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate parameters
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")
            if not (1 <= max_tokens <= 4096):
                raise ValueError("Max tokens must be between 1 and 4096")
            
            # Get audio duration for response (still need to process for duration)
            audio_array = self._process_audio_file(audio_file)
            audio_duration = len(audio_array) / 16000  # Assuming 16kHz
            
            # Create transcription prompt
            if language:
                text_prompt = f"Please transcribe this audio in {language}."
            else:
                text_prompt = "Please transcribe this audio."
            
            # Generate transcription using the file path directly
            transcription = await self._generate_response(text_prompt, audio_file, temperature, max_tokens)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return AudioResponse(
                text=transcription,
                duration_seconds=audio_duration,
                model_info={
                    "model_name": self.config.MODEL_NAME,
                    "context_length": str(self.config.MAX_MODEL_LEN),
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    @bentoml.api
    async def audio_qa(
        self,
        audio_file: Path,
        question: str,
        language: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AudioResponse:
        """Answer questions about audio content"""
        # Initialize engine if not already done
        if self.engine is None:
            await self._load_model()
            
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate parameters
            if not question.strip():
                raise ValueError("Question cannot be empty")
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")
            if not (1 <= max_tokens <= 4096):
                raise ValueError("Max tokens must be between 1 and 4096")
            
            # Get audio duration for response
            audio_array = self._process_audio_file(audio_file)
            audio_duration = len(audio_array) / 16000
            
            # Create Q&A prompt
            if language:
                text_prompt = f"Listen to this audio and answer the following question in {language}: {question}"
            else:
                text_prompt = f"Listen to this audio and answer the following question: {question}"
            
            # Generate response using the file path directly
            answer = await self._generate_response(text_prompt, audio_file, temperature, max_tokens)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return AudioResponse(
                text=answer,
                duration_seconds=audio_duration,
                model_info={
                    "model_name": self.config.MODEL_NAME,
                    "context_length": str(self.config.MAX_MODEL_LEN),
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Audio Q&A failed: {str(e)}")
            raise
    
    @bentoml.api
    async def health(self) -> Dict[str, str]:
        """Health check endpoint"""
        try:
            # Simple engine check
            if self.engine is None:
                return {"status": "unhealthy", "reason": "Engine not loaded"}
            
            return {
                "status": "healthy",
                "model": self.config.MODEL_NAME,
                "gpu_available": str(torch.cuda.is_available()),
                "gpu_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0"
            }
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}