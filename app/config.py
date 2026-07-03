from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    catalog_path: Path = ROOT_DIR / "shl_product_catalog.json"
    data_dir: Path = ROOT_DIR / "data"
    embedding_model: str = "all-MiniLM-L6-v2"

    llm_provider: str = "gemini"  # groq | gemini
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    retrieval_top_k: int = 25
    retrieval_candidate_k: int = 20
    max_recommendations: int = 10
    max_conversation_turns: int = 8
    chat_request_timeout_seconds: float = 30.0
    llm_request_timeout_seconds: float = 25.0
    llm_temperature: float = 0.0


settings = Settings()
