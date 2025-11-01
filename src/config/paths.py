"""
Project paths management.
Provides centralized path definitions following Cookiecutter structure.
"""

from pathlib import Path
from typing import Optional
from .config_loader import ConfigLoader


class ProjectPaths:
    """
    Manages all project paths following Cookiecutter ML structure.
    
    This class provides a single source of truth for all file paths,
    making it easy to reorganize or move the project structure.
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None, project_root: Optional[Path] = None):
        """
        Initialize project paths.
        
        Args:
            config: ConfigLoader instance. If None, creates new one.
            project_root: Path to project root. If None, inferred from config.
        """
        if config is None:
            config = ConfigLoader()
        
        if project_root is None:
            self.project_root = config.project_root
        else:
            self.project_root = Path(project_root).resolve()
        
        self.config = config
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup all project paths from configuration."""
        # Data paths
        self.raw_dir = self.project_root / self.config.get("data.raw_dir", "data/raw")
        self.interim_dir = self.project_root / self.config.get("data.interim_dir", "data/interim")
        self.processed_dir = self.project_root / self.config.get("data.processed_dir", "data/processed")
        
        # Model paths
        models_dir = self.config.get("paths.models_dir", "models")
        self.models_dir = self.project_root / models_dir
        
        # Report paths
        self.reports_dir = self.project_root / self.config.get("paths.reports_dir", "reports")
        self.figures_dir = self.project_root / self.config.get("paths.figures_dir", "reports/figures")
        
        # MLflow paths
        mlruns_dir = self.config.get("paths.mlruns_dir", "mlruns")
        self.mlruns_dir = self.project_root / mlruns_dir
        
        # Source paths
        self.src_dir = self.project_root / "src"
        
        # Config paths
        self.config_dir = self.project_root / "config"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.mlruns_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # Data file paths
    @property
    def raw_data_file(self) -> Path:
        """Path to raw data file."""
        filename = self.config.get("data.raw_filename", "bike_sharing_modified.csv")
        return self.raw_dir / filename
    
    @property
    def clean_data_file(self) -> Path:
        """Path to cleaned data file."""
        filename = self.config.get("data.clean_filename", "bike_sharing_clean.csv")
        return self.interim_dir / filename
    
    # Model file paths
    def model_file(self, model_name: str) -> Path:
        """Path to saved model file."""
        return self.models_dir / f"{model_name}.pkl"
    
    @property
    def scaler_file(self) -> Path:
        """Path to saved scaler."""
        return self.models_dir / "scaler.pkl"
    
    # Feature importance paths
    def feature_importance_file(self, model_name: str) -> Path:
        """Path to feature importance CSV."""
        return self.models_dir / f"{model_name}_feature_importance.csv"
    
    # Processed data paths
    def processed_file(self, split: str, normalized: bool = False) -> Path:
        """
        Path to processed data file.
        
        Args:
            split: Dataset split ("train", "val", "test", or "validation")
            normalized: Whether file is normalized (adds "_normalized" suffix)
        
        Returns:
            Path to processed data file
        """
        if split == "validation":
            split = "val"
        
        suffix = "_normalized" if normalized else ""
        filename = f"bike_sharing_features_{split}{suffix}.csv"
        return self.processed_dir / filename
    
    def __repr__(self) -> str:
        return f"ProjectPaths(project_root={self.project_root})"

