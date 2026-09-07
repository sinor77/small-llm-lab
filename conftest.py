"""
conftest.py — makes pytest discover the project packages from the repo root.
"""
import sys, os

# Add the project root to sys.path so all imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
