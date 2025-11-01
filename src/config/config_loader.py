"""
Configuration loader using YAML files.
Centralizes all configuration parameters for reproducibility.
"""

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required. Install it with: pip install pyyaml"
    )

from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Loads and manages project configuration from YAML files.
    
    This class ensures all configuration parameters are centralized and
    easily accessible throughout the project, improving reproducibility.
    
    Attributes:
        config: Dictionary containing all configuration parameters
        project_root: Path to the project root directory
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default
                        location at config/config.yaml relative to project root.
        """
        if config_path is None:
            # Default: config/config.yaml relative to this file
            self.project_root = Path(__file__).resolve().parent.parent.parent
            config_path = self.project_root / "config" / "config.yaml"
        else:
            self.project_root = config_path.resolve().parent.parent
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config key (e.g., "data.raw_dir")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config = ConfigLoader()
            >>> config.get("data.raw_dir")
            "data/raw"
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "data", "models", "mlflow")
        
        Returns:
            Dictionary containing section configuration
        """
        return self.config.get(section, {})
    
    @property
    def project_name(self) -> str:
        """Get project name from config."""
        return self.get("project_name", "bike_sharing_demand")
    
    @property
    def seed(self) -> int:
        """Get random seed for reproducibility."""
        return self.get("reproducibility.seed", 42)
    
    @property
    def n_jobs(self) -> int:
        """Get number of parallel jobs."""
        return self.get("reproducibility.n_jobs", -1)

