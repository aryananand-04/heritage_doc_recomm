"""
Configuration Loader Utility
Load and manage project configuration from YAML file
"""

import yaml
import os
from pathlib import Path

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._ensure_directories()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs_to_create = [
            *self._config['paths']['data'].values(),
            self._config['paths']['knowledge_graph']['base'],
            *self._config['paths']['models'].values(),
            self._config['paths']['results']['base'],
            self._config['paths']['results']['visualizations'],
            'logs'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get(self, *keys, default=None):
        """
        Get configuration value using dot notation
        Example: config.get('models', 'autoencoder', 'epochs')
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_path(self, *keys):
        """Get path from configuration"""
        return self.get('paths', *keys)
    
    def get_model_config(self, model_name):
        """Get model-specific configuration"""
        return self.get('models', model_name)
    
    @property
    def all(self):
        """Get entire configuration"""
        return self._config

# Global config instance
_config_instance = None

def get_config(config_path="config/config.yaml"):
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

# Example usage:
if __name__ == "__main__":
    config = get_config()
    
    print("Configuration loaded successfully!")
    print(f"\nEmbedding dimension: {config.get('models', 'sentence_transformer', 'embedding_dim')}")
    print(f"Number of clusters: {config.get('models', 'clustering', 'n_clusters')}")
    print(f"SimRank iterations: {config.get('models', 'simrank', 'max_iterations')}")
    print(f"\nKG path: {config.get_path('knowledge_graph', 'kg_file')}")
    
    # Test model config
    autoencoder_config = config.get_model_config('autoencoder')
    print(f"\nAutoencoder config: {autoencoder_config}")