#!/bin/bash

echo "========================================"
echo " Intelligent Document Search Setup"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.9+ first"
    exit 1
fi

echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/5] Creating data directories..."
mkdir -p data
mkdir -p chroma_db

echo "[4/5] Copying environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo
    echo "[IMPORTANT] Please edit .env file with your API keys:"
    echo "  - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys"
    echo "  - KAGGLE_USERNAME and KAGGLE_KEY: Get from https://kaggle.com/settings"
    echo
fi

echo "[5/5] Setup complete!"
echo
echo "========================================"
echo " Next Steps:"
echo "========================================"
echo "1. Edit .env file with your API keys"
echo "2. Download dataset: python backend/data_processor.py"
echo "3. Start backend: python backend/main.py"
echo "4. In new terminal, setup frontend:"
echo "   cd frontend"
echo "   npm install"
echo "   npm start"
echo
