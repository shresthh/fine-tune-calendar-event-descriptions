#!/usr/bin/env python3
"""
Resume training from checkpoint
"""

import os
import torch
import json
from pathlib import Path

from src.core.core.model_setup import EntityExtractionModel, ModelConfig, load_model
from src.core.core.custom_trainer import CustomTrainer
from src.core.core.data_analysis import DataAnalyzer, DataPreprocessor

def find_latest_experiment():
    """Find the most recent experiment directory"""
    exp_base = Path("experiments")
    if not exp_base.exists():
        # Look for test_outputs instead
        if Path("test_outputs").exists():
            return "test_outputs"
        else:
            print("No experiment directories found!")
            return None
    
    # Get all experiment directories
    exp_dirs = [d for d in exp_base.iterdir() if d.is_dir()]
    if not exp_dirs:
        print("No experiment directories found!")
        return None
    
    # Sort by modification time and get the latest
    latest_exp = max(exp_dirs, key=lambda d: d.stat().st_mtime)
    return str(latest_exp)

def find_latest_checkpoint(exp_dir):
    """Find the latest checkpoint in experiment directory"""
    checkpoint_dir = Path(exp_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        # Check if there are checkpoints in the experiment dir directly
        checkpoint_files = list(Path(exp_dir).glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            return str(latest_checkpoint)
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    def get_epoch_num(filename):
        try:
            return int(filename.stem.split('_')[-1])
        except:
            return 0
    
    latest_checkpoint = max(checkpoint_files, key=get_epoch_num)
    return str(latest_checkpoint)

def resume_from_checkpoint(checkpoint_path, remaining_epochs=2):
    """Resume training from checkpoint"""
    
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Fallback to default config
        config = ModelConfig()
    
    # Load data
    analyzer = DataAnalyzer('data/event_text_mapping.jsonl')
    train_data, val_data, test_data = analyzer.create_splits()
    
    # Preprocess data
    preprocessor = DataPreprocessor(config.model_name)
    train_processed = preprocessor.prepare_dataset(train_data, config.max_length)
    val_processed = preprocessor.prepare_dataset(val_data, config.max_length)
    
    # Setup model and trainer
    model = load_model(config)
    exp_dir = str(Path(checkpoint_path).parent.parent)
    trainer = CustomTrainer(model, config, exp_dir)
    
    # Load checkpoint state
    trainer.load_checkpoint(checkpoint_path)
    
    # Update number of epochs to continue training
    original_epochs = config.num_epochs
    trainer.config.num_epochs = trainer.current_epoch + remaining_epochs
    
    print(f"Resuming from epoch {trainer.current_epoch}")
    print(f"Will train for {remaining_epochs} more epochs (until epoch {trainer.config.num_epochs - 1})")
    
    # Continue training
    training_results = trainer.train(train_processed, val_processed)
    
    return training_results, exp_dir

def main():
    """Main resume script"""
    
    # Find latest experiment
    exp_dir = find_latest_experiment()
    if not exp_dir:
        print("No experiment directory found. Please run training first.")
        return
    
    print(f"Found experiment directory: {exp_dir}")
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(exp_dir)
    if not checkpoint_path:
        print(f"No checkpoint found in {exp_dir}")
        print("Available files:")
        for f in Path(exp_dir).rglob("*.pt"):
            print(f"  {f}")
        return
    
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Resume training
    try:
        training_results, final_exp_dir = resume_from_checkpoint(checkpoint_path, remaining_epochs=2)
        
        print("\n" + "="*60)
        print("TRAINING RESUMED AND COMPLETED!")
        print("="*60)
        print(f"Final results saved in: {final_exp_dir}")
        print(f"Best eval score: {training_results['best_eval_score']:.4f}")
        
    except Exception as e:
        print(f"Error resuming training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()