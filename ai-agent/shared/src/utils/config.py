"""
Configuration management for Indonesian Legal RAG System
"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "postgresql://user:pass@localhost:5432/legal_rag"
    redis_url: str = "redis://localhost:6379"
    vector_db_path: str = "data/vectors/vector_database.pkl"
    vector_db_url: str = "http://localhost:8000"
    max_connections: int = 20


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_period: int = 60


@dataclass
class LLMConfig:
    """LLM configuration"""
    default_provider: str = "deepseek"
    default_model: str = "deepseek-reasoner"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    
    # API keys (should be set via environment variables)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 500
    overlap_size: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    use_semantic_search: bool = True
    batch_size: int = 32
    
    # Scraping configuration
    scrape_interval_hours: int = 24
    max_scraping_workers: int = 4
    rate_limit_delay: float = 1.0


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    prometheus_port: int = 9090
    grafana_port: int = 3000
    log_level: str = "INFO"
    log_file: Optional[str] = None
    metrics_retention_days: int = 30


@dataclass
class Config:
    """Main configuration class"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Environment-specific settings
    environment: str = "development"
    debug: bool = False
    
    # Paths
    data_dir: str = "data"
    logs_dir: str = "data/logs"
    temp_dir: str = "data/temp"
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Load API keys from environment
        self.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.llm.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Override with environment variables
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Create directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        # Extract nested configurations
        database_data = data.get('database', {})
        api_data = data.get('api', {})
        llm_data = data.get('llm', {})
        processing_data = data.get('processing', {})
        monitoring_data = data.get('monitoring', {})
        
        # Create config objects
        config = cls()
        config.database = DatabaseConfig(**database_data)
        config.api = APIConfig(**api_data)
        config.llm = LLMConfig(**llm_data)
        config.processing = ProcessingConfig(**processing_data)
        config.monitoring = MonitoringConfig(**monitoring_data)
        
        # Set other attributes
        for key, value in data.items():
            if not hasattr(config, key) or key in ['database', 'api', 'llm', 'processing', 'monitoring']:
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'llm': self.llm.__dict__,
            'processing': self.processing.__dict__,
            'monitoring': self.monitoring.__dict__,
            'environment': self.environment,
            'debug': self.debug,
            'data_dir': self.data_dir,
            'logs_dir': self.logs_dir,
            'temp_dir': self.temp_dir
        }
    
    def save(self, config_path: str):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        return self.database.redis_url
    
    def get_vector_db_path(self) -> str:
        """Get vector database path"""
        return Path(self.data_dir) / "vectors" / "vector_database.pkl"
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider"""
        key_map = {
            'openai': self.llm.openai_api_key,
            'anthropic': self.llm.anthropic_api_key,
            'deepseek': self.llm.deepseek_api_key,
            'groq': self.llm.groq_api_key
        }
        return key_map.get(provider.lower())
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for provider"""
        return bool(self.get_api_key(provider))


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance"""
    global _config
    
    if _config is None:
        if config_path:
            _config = Config.from_file(config_path)
        else:
            # Try to load from default locations
            default_paths = [
                "config.yaml",
                "config.yml",
                "config.json",
                "config/config.yaml",
                "config/config.yml",
                "config/config.json"
            ]
            
            for path in default_paths:
                if Path(path).exists():
                    _config = Config.from_file(path)
                    break
            else:
                # Use default configuration
                _config = Config()
    
    return _config


def set_config(config: Config):
    """Set global configuration instance"""
    global _config
    _config = config


def load_config_from_env() -> Config:
    """Load configuration from environment variables"""
    config = Config()
    
    # Override with environment variables
    if os.getenv("DATABASE_URL"):
        config.database.url = os.getenv("DATABASE_URL")
    if os.getenv("REDIS_URL"):
        config.database.redis_url = os.getenv("REDIS_URL")
    if os.getenv("VECTOR_DB_PATH"):
        config.database.vector_db_path = os.getenv("VECTOR_DB_PATH")
    if os.getenv("VECTOR_DB_URL"):
        config.database.vector_db_url = os.getenv("VECTOR_DB_URL")
    
    if os.getenv("API_HOST"):
        config.api.host = os.getenv("API_HOST")
    if os.getenv("API_PORT"):
        config.api.port = int(os.getenv("API_PORT"))
    
    if os.getenv("DEFAULT_PROVIDER"):
        config.llm.default_provider = os.getenv("DEFAULT_PROVIDER")
    if os.getenv("DEFAULT_MODEL"):
        config.llm.default_model = os.getenv("DEFAULT_MODEL")
    
    if os.getenv("CHUNK_SIZE"):
        config.processing.chunk_size = int(os.getenv("CHUNK_SIZE"))
    if os.getenv("OVERLAP_SIZE"):
        config.processing.overlap_size = int(os.getenv("OVERLAP_SIZE"))
    if os.getenv("USE_SEMANTIC_SEARCH"):
        config.processing.use_semantic_search = os.getenv("USE_SEMANTIC_SEARCH").lower() == "true"
    
    if os.getenv("LOG_LEVEL"):
        config.monitoring.log_level = os.getenv("LOG_LEVEL")
    
    return config


# Configuration validation
def validate_config(config: Config) -> bool:
    """Validate configuration"""
    errors = []
    
    # Validate database configuration
    if not config.database.url:
        errors.append("Database URL is required")
    
    if not config.database.redis_url:
        errors.append("Redis URL is required")
    
    # Validate API configuration
    if config.api.port < 1 or config.api.port > 65535:
        errors.append("API port must be between 1 and 65535")
    
    # Validate processing configuration
    if config.processing.chunk_size <= 0:
        errors.append("Chunk size must be positive")
    
    if config.processing.overlap_size < 0:
        errors.append("Overlap size cannot be negative")
    
    if config.processing.overlap_size >= config.processing.chunk_size:
        errors.append("Overlap size must be less than chunk size")
    
    # Validate LLM configuration
    valid_providers = ['openai', 'anthropic', 'deepseek', 'groq']
    if config.llm.default_provider not in valid_providers:
        errors.append(f"Default provider must be one of: {valid_providers}")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        raise ValueError(error_msg)
    
    return True