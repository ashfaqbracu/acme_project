import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "embeddings": {
                "model_name": "shihab17/bangla-sentence-transformer",
                "db_path": "./vectorstore"
            },
            "llm": {
                "model_name": "microsoft/DialoGPT-medium",
                "max_length": 512,
                "temperature": 0.7
            },
            "data": {
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }

def save_config(config: Dict[str, Any], config_path: str = "config.json"):
    """Save configuration to JSON file"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def validate_environment():
    """Validate required environment setup"""
    required_dirs = ["data/raw", "data/processed", "vectorstore"]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("Environment validation complete")

def format_response(response: Dict[str, Any]) -> str:
    """Format response for display"""
    output = f"**Question:** {response['question']}\n\n"
    output += f"**Answer:** {response['answer']}\n\n"
    
    if response['citations']:
        output += "**Sources:**\n"
        for citation in response['citations']:
            output += f"- {citation['id']}: {citation['source']} ({citation['language']})\n"
    
    return output
