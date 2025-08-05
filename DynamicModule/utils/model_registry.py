import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from omegaconf import OmegaConf


class ModelRegistry:
    """Registry for managing different pretrained models and their configurations"""
    
    def __init__(self, registry_file: str = "configs/model_registry.json"):
        self.registry_file = registry_file
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry from JSON file"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            # Create default registry
            default_registry = {
                "models": {
                    "dynamic_module": {
                        "display_name": "DynamicModule",
                        "config_file": "configs/dynamic.yaml",
                        "ckpt_path": "ckpts/dynamic_module.pt",
                        "width": 512,
                        "height": 320,
                        "description": "Base dynamic module for video generation",
                        "tags": ["video", "dynamic", "camera_control"]
                    }
                },
                "model_groups": {
                    "cami2v": {
                        "display_name": "CamI2V Models",
                        "description": "Camera-to-Video models",
                        "models": ["dynamic_module"]
                    }
                }
            }
            self._save_registry(default_registry)
            return default_registry
    
    def _save_registry(self, registry: Dict[str, Any]):
        """Save the model registry to JSON file"""
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a new model"""
        self.models["models"][model_id] = model_info
        self._save_registry(self.models)
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information by ID"""
        return self.models["models"].get(model_id)
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        return self.models["models"]
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Get model configuration by loading the config file"""
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found in registry")
        
        config_file = model_info["config_file"]
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found")
        
        return OmegaConf.load(config_file)
    
    def validate_model(self, model_id: str) -> bool:
        """Validate that a model has all required files"""
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False
        
        # Check config file exists
        config_file = model_info.get("config_file")
        if not config_file or not os.path.exists(config_file):
            return False
        
        # Check checkpoint exists (optional)
        ckpt_path = model_info.get("ckpt_path")
        if ckpt_path and not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found for model {model_id}")
        
        return True
    
    def create_model_config_template(self, model_id: str, template_name: str = "base"):
        """Create a new model configuration from template"""
        template_path = f"configs/templates/{template_name}.yaml"
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template {template_path} not found")
        
        # Load template
        template = OmegaConf.load(template_path)
        
        # Create new config
        new_config_path = f"configs/{model_id}.yaml"
        OmegaConf.save(template, new_config_path)
        
        return new_config_path


class ModelConfigManager:
    """Manager for handling model configurations and compatibility"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def get_compatible_models(self, target_resolution: tuple = None, 
                            target_features: list = None) -> list:
        """Get models compatible with specified requirements"""
        compatible_models = []
        
        for model_id, model_info in self.registry.list_models().items():
            if not self.registry.validate_model(model_id):
                continue
            
            # Check resolution compatibility
            if target_resolution:
                model_width = model_info.get("width", 0)
                model_height = model_info.get("height", 0)
                if (model_width, model_height) != target_resolution:
                    continue
            
            # Check feature compatibility
            if target_features:
                model_tags = model_info.get("tags", [])
                if not any(feature in model_tags for feature in target_features):
                    continue
            
            compatible_models.append(model_id)
        
        return compatible_models
    
    def create_model_config(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """Create a model configuration with custom parameters"""
        base_config = self.registry.get_model_config(model_id)
        
        # Override with custom parameters
        for key, value in kwargs.items():
            if key in base_config:
                base_config[key] = value
        
        return base_config
    
    def validate_config_compatibility(self, config: Dict[str, Any]) -> bool:
        """Validate that a configuration is compatible with the system"""
        required_keys = ["model", "data"]
        
        for key in required_keys:
            if key not in config:
                return False
        
        # Check model target exists
        model_target = config["model"].get("target")
        if not model_target:
            return False
        
        # Check if target module exists (basic check)
        try:
            module_path, class_name = model_target.rsplit(".", 1)
            __import__(module_path)
        except ImportError:
            return False
        
        return True


def create_model_registry():
    """Create and initialize the model registry"""
    registry = ModelRegistry()
    
    # Register some example models
    example_models = {
        "dynamic_module": {
            "display_name": "DynamicModule",
            "config_file": "configs/dynamic.yaml",
            "ckpt_path": "ckpts/dynamic_module.pt",
            "width": 512,
            "height": 320,
            "description": "Dynamic module for 512x320 video generation",
            "tags": ["video", "dynamic", "camera_control"]
        },
    }
    
    for model_id, model_info in example_models.items():
        registry.register_model(model_id, model_info)
    
    return registry


if __name__ == "__main__":
    # Initialize registry
    registry = create_model_registry()
    manager = ModelConfigManager(registry)
    
    # List available models
    print("Available models:")
    for model_id, info in registry.list_models().items():
        print(f"  {model_id}: {info['display_name']}")
    
    # Get compatible models
    compatible = manager.get_compatible_models(target_resolution=(512, 320))
    print(f"\nModels compatible with 512x320: {compatible}") 