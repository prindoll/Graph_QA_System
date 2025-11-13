"""Configuration settings for GraphRAG"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    openai_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Graph Database Configuration
    graph_db_type: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Vector Store Configuration
    vector_store_type: str = "chroma"
    chroma_persist_directory: str = "./data/chroma_db"
    
    # Retrieval Configuration
    retrieval_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # General Settings
    log_level: str = "INFO"
    debug: bool = False
    batch_size: int = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
