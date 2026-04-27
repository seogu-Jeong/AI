import sys
import os
import shutil
from pathlib import Path

def get_app_dir():
    """Returns the directory where the frozen app's bundled read-only data lives or project root."""
    if getattr(sys, 'frozen', False):
        # PyInstaller bundled app
        return Path(sys._MEIPASS)
    # Development mode (root of the project)
    return Path(__file__).parent.absolute()

def get_user_data_dir():
    """Returns a writable directory for user data at ~/Library/Application Support/StockSense/."""
    # Note: Using Path.home() / "Library" / "Application Support" for macOS (darwin)
    data_dir = Path.home() / "Library" / "Application Support" / "StockSense"
    
    # Ensure necessary subdirectories exist
    subdirs = ["exports", "cache/images"]
    for sd in subdirs:
        (data_dir / sd).mkdir(parents=True, exist_ok=True)
    
    return data_dir

def get_resource(relative_path):
    """Joins get_app_dir() with relative_path for read-only bundled files."""
    return get_app_dir() / Path(relative_path)

def get_user_file(relative_path):
    """Joins get_user_data_dir() with relative_path for writable files."""
    return get_user_data_dir() / Path(relative_path)

def get_config_path():
    """
    Returns the user-writable config path.
    Copies default config.json from app dir if it doesn't exist in user data dir.
    """
    user_config = get_user_file("config.json")
    if not user_config.exists():
        default_config = get_resource("config.json")
        if default_config.exists():
            shutil.copy2(default_config, user_config)
    return user_config

def get_db_url():
    """Returns SQLAlchemy URL string for stocksense.db inside get_user_data_dir()."""
    db_path = get_user_file("stocksense.db").absolute()
    return f"sqlite:///{db_path}"

def get_weights_dir():
    """Returns absolute path to weights/ inside get_app_dir()."""
    return get_resource("weights")
