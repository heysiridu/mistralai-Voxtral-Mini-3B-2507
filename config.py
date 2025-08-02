# updated on Aug 1st 2025
import os
from typing import Optional

class VoxtralConfig:
    """Configuration class for Voxtral-Mini-3B-2507 model deployment"""
    
    # Model Configuration
    MODEL_NAME: str = "mistralai/Voxtral-Mini-3B-2507"
    MODEL_REVISION: Optional[str] = None
    
    # Context and Memory Settings
    MAX_MODEL_LEN: int = 32768  # 32k context length
    MAX_BATCH_SIZE: int = 4
    GPU_MEMORY_UTILIZATION: float = 0.85  # Use 85% of GPU memory
    
    # Audio Processing Settings
    AUDIO_SAMPLE_RATE: int = 16000  # 16kHz sampling rate
    MAX_AUDIO_LENGTH_SECONDS: int = 2400  # 40 minutes max (32k tokens ~ 30-40 min)
    SUPPORTED_AUDIO_FORMATS: list = [
        "wav", "mp3", "m4a", "flac", "aac", "ogg", "wma"
    ]
    
    # Generation Parameters
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 50
    REPETITION_PENALTY: float = 1.1
    MAX_TOKENS_TRANSCRIPTION: int = 1024
    MAX_TOKENS_QA: int = 2048
    
    # Performance Settings
    ENABLE_CHUNKED_PREFILL: bool = True
    MAX_NUM_BATCHED_TOKENS: int = 4096
    BLOCK_SIZE: int = 16
    SWAP_SPACE: int = 4  # GB
    
    # Timeout Settings
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    MODEL_LOAD_TIMEOUT: int = 600  # 10 minutes
    
    # Environment Variables (can override defaults)
    def __init__(self):
        # Model settings from environment
        self.MODEL_NAME = os.getenv("VOXTRAL_MODEL_NAME", self.MODEL_NAME)
        self.MODEL_REVISION = os.getenv("VOXTRAL_MODEL_REVISION", self.MODEL_REVISION)
        
        # Memory and performance settings
        self.MAX_MODEL_LEN = int(os.getenv("VOXTRAL_MAX_MODEL_LEN", self.MAX_MODEL_LEN))
        self.MAX_BATCH_SIZE = int(os.getenv("VOXTRAL_MAX_BATCH_SIZE", self.MAX_BATCH_SIZE))
        self.GPU_MEMORY_UTILIZATION = float(os.getenv("VOXTRAL_GPU_MEMORY_UTIL", self.GPU_MEMORY_UTILIZATION))
        
        # Audio processing settings
        self.AUDIO_SAMPLE_RATE = int(os.getenv("VOXTRAL_SAMPLE_RATE", self.AUDIO_SAMPLE_RATE))
        self.MAX_AUDIO_LENGTH_SECONDS = int(os.getenv("VOXTRAL_MAX_AUDIO_LENGTH", self.MAX_AUDIO_LENGTH_SECONDS))
        
        # Generation parameters
        self.DEFAULT_TEMPERATURE = float(os.getenv("VOXTRAL_TEMPERATURE", self.DEFAULT_TEMPERATURE))
        self.DEFAULT_TOP_P = float(os.getenv("VOXTRAL_TOP_P", self.DEFAULT_TOP_P))
        self.DEFAULT_TOP_K = int(os.getenv("VOXTRAL_TOP_K", self.DEFAULT_TOP_K))
        self.REPETITION_PENALTY = float(os.getenv("VOXTRAL_REP_PENALTY", self.REPETITION_PENALTY))
        
        # Timeout settings
        self.REQUEST_TIMEOUT = int(os.getenv("VOXTRAL_REQUEST_TIMEOUT", self.REQUEST_TIMEOUT))
        self.MODEL_LOAD_TIMEOUT = int(os.getenv("VOXTRAL_MODEL_LOAD_TIMEOUT", self.MODEL_LOAD_TIMEOUT))
        
        # Performance tuning
        self.ENABLE_CHUNKED_PREFILL = os.getenv("VOXTRAL_CHUNKED_PREFILL", "true").lower() == "true"
        self.MAX_NUM_BATCHED_TOKENS = int(os.getenv("VOXTRAL_MAX_BATCHED_TOKENS", self.MAX_NUM_BATCHED_TOKENS))
        self.BLOCK_SIZE = int(os.getenv("VOXTRAL_BLOCK_SIZE", self.BLOCK_SIZE))
        self.SWAP_SPACE = int(os.getenv("VOXTRAL_SWAP_SPACE", self.SWAP_SPACE))
    
    @property
    def TOP_P(self) -> float:
        """Get top_p value for sampling"""
        return self.DEFAULT_TOP_P
    
    @property
    def TOP_K(self) -> int:
        """Get top_k value for sampling"""
        return self.DEFAULT_TOP_K
    
    def get_vllm_engine_args(self) -> dict:
        """Get vLLM engine arguments"""
        return {
            "model": self.MODEL_NAME,
            "revision": self.MODEL_REVISION,
            "tokenizer": self.MODEL_NAME,
            "tokenizer_revision": self.MODEL_REVISION,
            "trust_remote_code": True,
            "max_model_len": self.MAX_MODEL_LEN,
            "gpu_memory_utilization": self.GPU_MEMORY_UTILIZATION,
            "swap_space": self.SWAP_SPACE,
            "block_size": self.BLOCK_SIZE,
            "max_num_batched_tokens": self.MAX_NUM_BATCHED_TOKENS,
            "enable_chunked_prefill": self.ENABLE_CHUNKED_PREFILL,
            "max_num_seqs": self.MAX_BATCH_SIZE,
            "enforce_eager": True,  # Required for multimodal
            "limit_mm_per_prompt": {"audio": 1},
            "disable_log_stats": False,
        }
    
    def validate_audio_duration(self, duration_seconds: float) -> bool:
        """Check if audio duration is within limits"""
        return duration_seconds <= self.MAX_AUDIO_LENGTH_SECONDS
    
    def validate_audio_format(self, file_extension: str) -> bool:
        """Check if audio format is supported"""
        return file_extension.lower().lstrip('.') in self.SUPPORTED_AUDIO_FORMATS
    
    def get_model_info(self) -> dict:
        """Get model information for responses"""
        return {
            "model_name": self.MODEL_NAME,
            "model_revision": self.MODEL_REVISION or "main",
            "max_context_length": self.MAX_MODEL_LEN,
            "max_audio_length_seconds": self.MAX_AUDIO_LENGTH_SECONDS,
            "supported_formats": self.SUPPORTED_AUDIO_FORMATS,
            "audio_sample_rate": self.AUDIO_SAMPLE_RATE,
        }

# Global configuration instance
config = VoxtralConfig()
