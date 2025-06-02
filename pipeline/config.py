import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    
from src.config import *  # Import all configs from src.config for consistency

# You can add pipeline-specific configs or logging setup here later
