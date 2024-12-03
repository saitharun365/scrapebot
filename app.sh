#!/bin/bash

# Run the scraping script
python3 embeddings.py

# Run the FastAPI server
python3 main.py