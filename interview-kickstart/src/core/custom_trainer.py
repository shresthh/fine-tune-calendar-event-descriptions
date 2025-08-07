#!/usr/bin/env python3
"""
Custom PyTorch Trainer for Entity Extraction Fine-tuning
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import wandb
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from src.core.core.model_setup import EntityExtractionModel, ModelConfig, EntityEvaluator
from src.core.core.data_analysis import DataPreprocessor

class EntityDataset(Dataset):
    """PyTorch Dataset for entity extraction training"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class CustomTrainer:
    """Custom PyTorch trainer for entity extraction"""
    
    def __init__(self, model: EntityExtractionModel, config: ModelConfig, 
                 output_dir: str = "outputs"):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.evaluator = EntityEvaluator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_score = 0.0
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_exact_match': [],
            'learning_rate': []
        }
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Trainer initialized. Using device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Separate parameters for different learning rates if needed
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.warmup_steps,
            T_mult=2,
            eta_min=self.config.learning_rate * 0.1
        )
    
    def save_checkpoint(self, epoch: int, eval_score: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'eval_score': eval_score,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with score: {eval_score:.4f}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        self.best_eval_score = checkpoint['eval_score']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train_epoch(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_base_loss = 0.0
        total_entity_loss = 0.0
        num_batches = len(train_dataloader)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            base_loss = outputs.get('base_loss', torch.tensor(0.0))
            entity_loss = outputs.get('entity_loss', torch.tensor(0.0))
            
            # Backward pass
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_base_loss += base_loss.item()
            total_entity_loss += entity_loss.item()
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log training step
            if self.global_step % 50 == 0:
                self.training_history['learning_rate'].append(current_lr)
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_base_loss = total_base_loss / num_batches
        avg_entity_loss = total_entity_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_base_loss': avg_base_loss,
            'train_entity_loss': avg_entity_loss
        }
    
    def evaluate(self, eval_dataloader: DataLoader, eval_samples: List[Dict]) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(eval_dataloader)
        
        # Calculate loss on evaluation set
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs['loss'].item()
        
        avg_loss = total_loss / num_batches
        
        # Evaluate entity extraction performance
        entity_metrics = self.evaluator.evaluate_batch(self.model, eval_samples[:50])  # Sample for speed
        
        metrics = {
            'eval_loss': avg_loss,
            **entity_metrics
        }
        
        return metrics
    
    def train(self, train_dataset: List[Dict], eval_dataset: List[Dict]) -> Dict[str, Any]:
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Eval samples: {len(eval_dataset)}")
        
        # Create data loaders (use num_workers=0 to avoid tokenizer parallelism issues)
        train_dataloader = DataLoader(
            EntityDataset(train_dataset),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Changed from 2 to 0 to avoid tokenizer warnings
            collate_fn=collate_fn
        )
        
        eval_dataloader = DataLoader(
            EntityDataset(eval_dataset),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Changed from 2 to 0 to avoid tokenizer warnings
            collate_fn=collate_fn
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            
            # Evaluate
            eval_metrics = self.evaluate(eval_dataloader, eval_dataset)
            self.training_history['eval_loss'].append(eval_metrics['eval_loss'])
            self.training_history['eval_exact_match'].append(eval_metrics['exact_match'])
            
            # Log metrics
            self.logger.info(f"Epoch {epoch} completed:")
            self.logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")
            self.logger.info(f"  Exact Match: {eval_metrics['exact_match']:.4f}")
            
            # Field-wise accuracy
            for field in ['action', 'date', 'time', 'location', 'duration']:
                if field in eval_metrics:
                    self.logger.info(f"  {field.capitalize()} Accuracy: {eval_metrics[field]:.4f}")
            
            # Save checkpoint
            eval_score = eval_metrics['exact_match']
            is_best = eval_score > self.best_eval_score
            if is_best:
                self.best_eval_score = eval_score
            
            self.save_checkpoint(epoch, eval_score, is_best)
        
        # Save final results
        self.save_training_plots()
        
        return {
            'best_eval_score': self.best_eval_score,
            'training_history': self.training_history,
            'final_metrics': eval_metrics
        }
    
    def save_training_plots(self):
        """Save training progress plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['eval_loss'], label='Eval Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Exact match accuracy
        axes[0, 1].plot(self.training_history['eval_exact_match'], label='Exact Match', color='green')
        axes[0, 1].set_title('Exact Match Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if self.training_history['learning_rate']:
            axes[1, 0].plot(self.training_history['learning_rate'], color='red')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # Loss difference
        if len(self.training_history['train_loss']) == len(self.training_history['eval_loss']):
            loss_diff = np.array(self.training_history['eval_loss']) - np.array(self.training_history['train_loss'])
            axes[1, 1].plot(loss_diff, label='Eval - Train Loss', color='orange')
            axes[1, 1].set_title('Overfitting Monitor')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to {self.output_dir}/training_plots.png")

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    try:
        # Extract tensors and other data
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Add non-tensor data
        for key in ['original_input', 'original_output', 'formatted_input', 'formatted_output']:
            if key in batch[0]:
                result[key] = [item[key] for item in batch]
        
        return result
    
    except RuntimeError as e:
        # If stacking fails due to size mismatch, print debug info and re-raise
        print(f"Collate function error: {e}")
        print(f"Batch size: {len(batch)}")
        if batch:
            print(f"Sample tensor shapes:")
            for i, item in enumerate(batch[:3]):  # Show first 3 items
                print(f"  Item {i}: input_ids shape = {item['input_ids'].shape}")
                print(f"  Item {i}: attention_mask shape = {item['attention_mask'].shape}")
                print(f"  Item {i}: labels shape = {item['labels'].shape}")
        raise e

def main():
    """Test custom trainer"""
    from src.core.core.data_analysis import DataAnalyzer, DataPreprocessor
    
    # Load and prepare data
    analyzer = DataAnalyzer('data/event_text_mapping.jsonl')
    train_data, val_data, test_data = analyzer.create_splits()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    train_processed = preprocessor.prepare_dataset(train_data[:100])  # Small subset for testing
    val_processed = preprocessor.prepare_dataset(val_data[:20])
    
    # Setup model and trainer
    config = ModelConfig(
        batch_size=2,
        num_epochs=3,
        learning_rate=1e-4
    )
    
    from src.core.core.model_setup import load_model
    model = load_model(config)
    
    trainer = CustomTrainer(model, config, output_dir="test_outputs")
    
    # Train
    results = trainer.train(train_processed, val_processed)
    
    print("Training completed!")
    print(f"Best eval score: {results['best_eval_score']:.4f}")

if __name__ == "__main__":
    main()