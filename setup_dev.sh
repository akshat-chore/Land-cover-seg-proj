#!/bin/bash

# Setup script for local development
# Run: bash setup_dev.sh

set -e

echo "🚀 Setting up Land-Cover Segmentation API..."

# Check Python version
echo "✓ Checking Python version..."
python3 --version

# Create virtual environment
echo "✓ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "✓ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "✓ Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "✓ Creating necessary directories..."
mkdir -p workdir logs output/predicted_masks output/prediction_plots

# Setup environment file
echo "✓ Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env file (please update GEMINI_API_KEY)"
else
    echo "  .env file already exists"
fi

# Run tests
echo "✓ Running tests..."
pytest tests/ -v --tb=short || true

# Summary
echo ""
echo "================================"
echo "✅ Setup completed successfully!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Update .env with your GEMINI_API_KEY"
echo "2. Start the server:"
echo "   python -m uvicorn app.main:app --reload"
echo "3. Open http://localhost:8000/docs in your browser"
echo ""
echo "To deactivate virtual environment: deactivate"
echo ""
