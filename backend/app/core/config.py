"""
Core configuration and settings for the Knowledge Chatbot backend.
"""
import os
import json
from pathlib import Path
from typing import List, Union, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from the project root
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    model_config = SettingsConfigDict(
        env_file="../.env",
        case_sensitive=True,
        env_prefix="",
        json_schema_extra={"env_parse_none_str": ""}
    )
    
    # Application
    APP_NAME: str = "Knowledge Chatbot API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "RAG-powered chatbot with streaming responses"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000"
    ]
    
    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse ALLOWED_ORIGINS from various formats."""
        if isinstance(v, str):
            # Handle empty string
            if not v or v.strip() == "":
                return ["http://localhost:5173", "http://localhost:3000"]
            # Try JSON parsing first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Fall back to comma-separated
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        elif isinstance(v, list):
            return v
        return ["http://localhost:5173", "http://localhost:3000"]
    
    # Provider selection
    PROVIDER: str = "ollama"  # "gemini" or "ollama"

    # Gemini (Google Generative AI)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"

    # Ollama (Local LLM)
    OLLAMA_BASE_URL: str = ""
    OLLAMA_MODEL: str = "mistral"  # e.g., "mistral", "llama2", "neural-chat", etc.
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TEMPERATURE: float = 0.7

    # Dimensionality reduction (optional)
    PCA: int = 768
    
    # Pinecone
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "knowra-onboarding-gemini"
    
    # RAG Configuration
    DATA_DIR: str = "../data/raw"
    VECTOR_STORE_DIR: str = "./backend/.vectorstore"  # For hash storage only
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVER_K: int = 5
    MEMORY_WINDOW: int = 10
    AUTO_REBUILD_ENABLED: bool = True  # Disable auto-rebuild on startup and chat
    
    # Advanced Chunking Options
    CHUNK_ADD_SECTION_HEADERS: bool = True  # Prepend section context to chunks
    CHUNK_EXTRACT_METADATA: bool = True     # Extract frontmatter and metadata
    
    # MongoDB Configuration
    MONGODB_URL: str = ""
    MONGODB_DB_NAME: str = "knowra_chatbot"
    MONGODB_MAX_POOL_SIZE: int = 10
    MONGODB_MIN_POOL_SIZE: int = 1
    MONGODB_TIMEOUT_MS: int = 5000
    
    # Redis Configuration (for session metadata)
    REDIS_URL: Optional[str] = None
    
    # SSE Streaming
    SSE_DELAY: float = 0.01


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings: Application settings
    """
    return settings
