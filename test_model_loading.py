#!/usr/bin/env python
"""Test script to debug model loading issues."""

import sys
import logging
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

print("\n" + "="*60)
print("Testing Model Loading")
print("="*60 + "\n")

try:
    logger.info("Step 1: Importing app.model module...")
    from app.model import SegmentationModel
    logger.info("✓ Successfully imported SegmentationModel")
    
    logger.info("\nStep 2: Creating SegmentationModel instance...")
    model = SegmentationModel(
        checkpoint_path='models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth',
        encoder='efficientnet-b0',
        encoder_weights='imagenet',
        num_classes=5,
        device='cpu'
    )
    logger.info("✓ SegmentationModel created successfully")
    logger.info(f"  Device: {model.device}")
    logger.info(f"  Model type: {type(model.model)}")
    
    print("\n" + "="*60)
    print("✓ MODEL LOADING SUCCESSFUL!")
    print("="*60)
    
except Exception as e:
    import traceback
    print("\n" + "="*60)
    print("✗ MODEL LOADING FAILED!")
    print("="*60)
    logger.error(f"Error: {e}")
    logger.error(f"\nFull traceback:\n{traceback.format_exc()}")
    sys.exit(1)
