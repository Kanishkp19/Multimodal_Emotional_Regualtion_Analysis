#!/bin/bash

# Emotion Regulation Analysis - Quick Start Script
# This script runs the entire pipeline from start to finish

set -e  # Exit on error

echo "================================"
echo "Emotion Regulation Analysis"
echo "Quick Start Pipeline"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Virtual environment not activated"
    echo "Activating virtual environment..."
    
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        print_error "Virtual environment 'venv' not found!"
        echo "Please run: python3 -m venv venv && source venv/bin/activate"
        exit 1
    fi
fi

# Step 1: Check dependencies
print_step "1" "Checking dependencies..."
if ! python -c "import torch, transformers, librosa" 2>/dev/null; then
    print_warning "Dependencies not fully installed"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi
echo "✓ Dependencies OK"
echo ""

# Step 2: Download datasets
print_step "2" "Downloading datasets..."

# Check Kaggle credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    print_error "Kaggle credentials not found!"
    echo "Please setup Kaggle API:"
    echo "  1. Download kaggle.json from https://www.kaggle.com/settings"
    echo "  2. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  3. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Download IEMOCAP
if [ ! -d "data/iemocap" ] || [ -z "$(ls -A data/iemocap 2>/dev/null)" ]; then
    echo "Downloading IEMOCAP..."
    python preprocessing/download_iemocap.py
else
    echo "✓ IEMOCAP already downloaded"
fi

# Download MOSEI
if [ ! -d "data/mosei" ] || [ -z "$(ls -A data/mosei 2>/dev/null)" ]; then
    echo "Downloading MOSEI..."
    python preprocessing/download_mosei.py
else
    echo "✓ MOSEI already downloaded"
fi
echo ""

# Step 3: Extract features
print_step "3" "Extracting features..."

if [ ! -f "data/processed/iemocap_features.pkl" ]; then
    echo "Extracting IEMOCAP features..."
    python preprocessing/extract_features.py --dataset iemocap
else
    echo "✓ IEMOCAP features already extracted"
fi

if [ ! -f "data/processed/mosei_features.pkl" ]; then
    echo "Extracting MOSEI features..."
    python preprocessing/extract_features.py --dataset mosei
else
    echo "✓ MOSEI features already extracted"
fi
echo ""

# Step 4: Create datasets
print_step "4" "Creating processed datasets..."

if [ ! -f "data/processed/iemocap_train.pkl" ]; then
    python preprocessing/create_dataset.py --dataset both
else
    echo "✓ Processed datasets already created"
fi
echo ""

# Step 5: Train incongruence encoder
print_step "5" "Training incongruence encoder..."

if [ ! -f "checkpoints/incongruence_encoder_best.pth" ]; then
    echo "Training incongruence encoder (this may take 20-40 minutes)..."
    python training/train_incongruence.py --epochs 15 --batch_size 32 --lr 0.001
else
    echo "✓ Incongruence encoder already trained"
fi
echo ""

# Step 6: Train temporal model
print_step "6" "Training temporal model..."

if [ ! -f "checkpoints/temporal_model_best.pth" ]; then
    echo "Training temporal model (this may take 30-50 minutes)..."
    python training/train_temporal.py --epochs 20 --batch_size 32 --lr 0.001
else
    echo "✓ Temporal model already trained"
fi
echo ""

# Step 7: Evaluate
print_step "7" "Evaluating models..."

if [ ! -f "results/eci_scores_iemocap.csv" ]; then
    python evaluate.py --dataset both
else
    echo "✓ Evaluation already completed"
fi
echo ""

# Done!
echo "================================"
echo -e "${GREEN}✓ Pipeline Complete!${NC}"
echo "================================"
echo ""
echo "Results available at:"
echo "  - results/eci_scores_iemocap.csv"
echo "  - results/eci_scores_mosei.csv"
echo "  - results/cross_dataset_comparison.json"
echo ""
echo "Next steps:"
echo "  1. View results: cat results/summary_statistics.csv"
echo "  2. Run analysis: jupyter notebook analysis/visualization.ipynb"
echo "  3. Check training logs: tensorboard --logdir logs"
echo ""