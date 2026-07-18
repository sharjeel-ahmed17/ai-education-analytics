from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # Relational Database URI (default to SQLite if not provided)
    database_url: Optional[str] = None
    
    # LLM API Keys
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    
    # Qdrant Vector Store
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    
    # Streamlit Configs
    streamlit_server_port: int = 8501
    streamlit_server_address: str = "0.0.0.0"

settings = Settings()
