"""Configuration settings for GraphRAG"""
from typing import Any, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    openai_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    
    # Embedding Configuration
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Graph Database Configuration
    graph_db_type: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # Vector Store Configuration
    vector_store_type: str = "chroma"
    chroma_persist_directory: str = "./data/chroma_db"
    
    # Retrieval Configuration
    retrieval_mode_default: str = "auto"
    retrieval_top_k: int = 5
    max_hops: int = 2
    chunk_size: int = 1200
    chunk_overlap: int = 100
    community_algorithm: str = "leiden"
    
    # General Settings
    log_level: str = "INFO"
    debug: bool = False
    batch_size: int = 32

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "t", "yes", "y", "on", "debug"}:
                return True
            if normalized in {"0", "false", "f", "no", "n", "off", "release", "prod", "production"}:
                return False
        return bool(value)
    

# Global settings instance
settings = Settings()
