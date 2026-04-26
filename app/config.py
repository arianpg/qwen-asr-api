from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    default_model: str = "Qwen/Qwen3-ASR-1.7B"
    model_unload_timeout: int = 300
    device: str = "cuda"
    enable_timestamps: bool = False
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"


settings = Settings()
