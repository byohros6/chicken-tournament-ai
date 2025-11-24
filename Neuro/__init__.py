import sys
import os

# --- AUTO-DETECT GAME ENGINE ---
# Get the path to "Dist/3600-agents/Neuro/"
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to "Dist/"
dist_dir = os.path.dirname(os.path.dirname(current_dir))
# Build path to "Dist/engine"
engine_dir = os.path.join(dist_dir, 'engine')

# Add to Python path if not already there
if engine_dir not in sys.path:
    sys.path.append(engine_dir)

# Now we can safely import
from .agent import PlayerAgent