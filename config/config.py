# config/config.py
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    AZURE_OPENAI_EMBEDDING_ENDPOINT: str = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    AZURE_OPENAI_EMBEDDING_API_VERSION: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
    
    # Tavily Configuration
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    
    # Application Settings
    MAX_HISTORY_LENGTH: int = 20
    VECTOR_DB_PATH: str = "./vector_db"
    DOCUMENTS_PATH: str = "./documents"
    
    # Response Settings
    CONCISE_MAX_TOKENS: int = 300
    DETAILED_MAX_TOKENS: int = 2000
    
    # Medical Research Specific Settings
    MEDICAL_TERM_THRESHOLD: float = 0.7  # Confidence threshold for medical term detection
    
    def validate(self) -> bool:
        """Validate that required API keys are present"""
        required_fields = [
            self.AZURE_OPENAI_ENDPOINT,
            self.AZURE_OPENAI_API_KEY,
            self.TAVILY_API_KEY
        ]
        return all(field.strip() for field in required_fields)

config = Config()