import os, sys

# Use modules from src directory in notebooks
path = os.path.dirname(os.path.realpath(__file__)) + '/../src'
sys.path.insert(1, path)