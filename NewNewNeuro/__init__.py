import sys
import os

# Auto-detect 'engine' folder relative to this file
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
engine_path = os.path.join(grandparent, 'engine')

if engine_path not in sys.path:
    sys.path.append(engine_path)