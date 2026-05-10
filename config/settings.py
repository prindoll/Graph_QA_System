"""Runtime settings."""
from typing import Any, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")
    
    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    openai_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_request_timeout_seconds: int = 120
    llm_extraction_max_tokens: int = 900
    llm_community_report_max_tokens: int = 900
    llm_extraction_text_chars: int = 3500
    max_llm_community_reports: int = 25
    min_llm_community_size: int = 3
    ollama_base_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "5m"
    
    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Graph database
    graph_db_type: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # Vector store
    vector_store_type: str = "chroma"
    chroma_persist_directory: str = "./data/chroma_db"
    
    # Retrieval
    retrieval_mode_default: str = "auto"
    retrieval_top_k: int = 5
    max_hops: int = 2
    chunk_size: int = 1200
    chunk_overlap: int = 100
    index_max_concurrent: int = 6
    index_progress_heartbeat_seconds: int = 15
    index_slow_chunk_seconds: int = 20
    hybrid_candidate_multiplier: int = 4
    hybrid_vector_weight: float = 0.65
    hybrid_keyword_weight: float = 0.25
    hybrid_graph_weight: float = 0.10
    community_algorithm: str = "leiden"
    
    # App
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
    

settings = Settings()
