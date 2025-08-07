#!/usr/bin/env python3
"""
Main Training Script for Entity Extraction Fine-tuning
"""

import os
import json
import torch
import argparse
from datetime import datetime
from typing import Dict, List, Any

from src.core.data_analysis import DataAnalyzer, DataPreprocessor
from src.core.model_setup import EntityExtractionModel, ModelConfig, load_model
from src.core.custom_trainer import CustomTrainer, collate_fn
from src.evaluation.baseline_evaluation import BaselineEvaluator, create_evaluation_plots

def setup_experiment(experiment_name: str = None) -> str:
    """Setup experiment directory and logging"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"entity_extraction_{timestamp}"
    
    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['checkpoints', 'plots', 'results']:
        os.makedirs(f"{exp_dir}/{subdir}", exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")
    return exp_dir

def run_baseline_evaluation(test_data: List[Dict], exp_dir: str) -> Dict[str, Any]:
    """Run baseline model evaluation"""
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)
    
    config = ModelConfig()
    baseline_model = load_model(config)
    
    evaluator = BaselineEvaluator(baseline_model)
    baseline_results = evaluator.evaluate_comprehensive(
        test_data, 
        f"{exp_dir}/results/baseline_results.json"
    )
    
    evaluator.print_summary(baseline_results)
    create_evaluation_plots(baseline_results, f"{exp_dir}/plots/baseline")
    
    return baseline_results

def run_fine_tuning(train_data: List[Dict], val_data: List[Dict], 
                   config: ModelConfig, exp_dir: str) -> Dict[str, Any]:
    """Run fine-tuning process"""
    print("\n" + "="*60)
    print("FINE-TUNING PROCESS")
    print("="*60)
    
    # Prepare data
    preprocessor = DataPreprocessor(config.model_name)
    
    print("Preprocessing training data...")
    train_processed = preprocessor.prepare_dataset(train_data, config.max_length)
    print(f"Train samples processed: {len(train_processed)}")
    
    print("Preprocessing validation data...")
    val_processed = preprocessor.prepare_dataset(val_data, config.max_length)
    print(f"Validation samples processed: {len(val_processed)}")
    
    # Setup model and trainer
    model = load_model(config)
    trainer = CustomTrainer(model, config, f"{exp_dir}/checkpoints")
    
    # Start training
    training_results = trainer.train(train_processed, val_processed)
    
    return training_results

def run_final_evaluation(test_data: List[Dict], exp_dir: str, 
                        baseline_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluation of fine-tuned model and compare with baseline"""
    print("\n" + "="*60)
    print("FINE-TUNED MODEL EVALUATION")
    print("="*60)
    
    # Load best fine-tuned model
    config = ModelConfig()
    model = load_model(config)
    
    best_checkpoint_path = f"{exp_dir}/checkpoints/best_model.pt"
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_checkpoint_path}")
    else:
        print("Warning: No fine-tuned model found, using baseline model")
    
    # Evaluate fine-tuned model
    evaluator = BaselineEvaluator(model)
    finetuned_results = evaluator.evaluate_comprehensive(
        test_data, 
        f"{exp_dir}/results/finetuned_results.json"
    )
    
    evaluator.print_summary(finetuned_results)
    create_evaluation_plots(finetuned_results, f"{exp_dir}/plots/finetuned")
    
    # Compare results
    comparison = compare_results(baseline_results, finetuned_results)
    
    # Save comparison
    with open(f"{exp_dir}/results/comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print_comparison(comparison)
    
    return finetuned_results

def compare_results(baseline_results: Dict[str, Any], 
                   finetuned_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline and fine-tuned model results"""
    
    baseline_metrics = baseline_results['metrics']
    finetuned_metrics = finetuned_results['metrics']
    
    comparison = {
        'exact_match_improvement': finetuned_metrics['exact_match_accuracy'] - baseline_metrics['exact_match_accuracy'],
        'json_quality_improvement': finetuned_metrics['valid_json_rate'] - baseline_metrics['valid_json_rate'],
        'completeness_improvement': finetuned_metrics['complete_fields_rate'] - baseline_metrics['complete_fields_rate'],
        'field_improvements': {},
        'baseline_scores': {
            'exact_match': baseline_metrics['exact_match_accuracy'],
            'valid_json_rate': baseline_metrics['valid_json_rate'],
            'complete_fields_rate': baseline_metrics['complete_fields_rate']
        },
        'finetuned_scores': {
            'exact_match': finetuned_metrics['exact_match_accuracy'],
            'valid_json_rate': finetuned_metrics['valid_json_rate'],
            'complete_fields_rate': finetuned_metrics['complete_fields_rate']
        }
    }
    
    # Field-wise improvements
    for field in baseline_metrics['field_accuracy']:
        baseline_acc = baseline_metrics['field_accuracy'][field]
        finetuned_acc = finetuned_metrics['field_accuracy'][field]
        comparison['field_improvements'][field] = finetuned_acc - baseline_acc
    
    return comparison

def print_comparison(comparison: Dict[str, Any]):
    """Print comparison results"""
    print("\n" + "="*60)
    print("BASELINE vs FINE-TUNED COMPARISON")
    print("="*60)
    
    print(f"\nOverall Improvements:")
    print(f"  Exact Match:      {comparison['exact_match_improvement']:+.3f}")
    print(f"  JSON Quality:     {comparison['json_quality_improvement']:+.3f}")
    print(f"  Completeness:     {comparison['completeness_improvement']:+.3f}")
    
    print(f"\nField-wise Improvements:")
    for field, improvement in comparison['field_improvements'].items():
        sign = "+" if improvement >= 0 else ""
        print(f"  {field:<12}: {sign}{improvement:.3f}")
    
    print(f"\nAbsolute Scores:")
    print(f"  Metric            Baseline    Fine-tuned   Improvement")
    print(f"  ---------------   --------    ----------   -----------")
    
    baseline = comparison['baseline_scores']
    finetuned = comparison['finetuned_scores']
    
    for metric in ['exact_match', 'valid_json_rate', 'complete_fields_rate']:
        b_score = baseline[metric]
        f_score = finetuned[metric]
        improvement = f_score - b_score
        sign = "+" if improvement >= 0 else ""
        print(f"  {metric:<15}   {b_score:.3f}       {f_score:.3f}        {sign}{improvement:.3f}")

def save_experiment_summary(exp_dir: str, config: ModelConfig, 
                           baseline_results: Dict[str, Any],
                           training_results: Dict[str, Any],
                           finetuned_results: Dict[str, Any]):
    """Save comprehensive experiment summary"""
    
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'model_name': config.model_name,
            'max_length': config.max_length,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'entity_loss_weight': config.entity_loss_weight
        },
        'baseline_performance': {
            'exact_match_accuracy': baseline_results['metrics']['exact_match_accuracy'],
            'valid_json_rate': baseline_results['metrics']['valid_json_rate'],
            'field_accuracy': baseline_results['metrics']['field_accuracy']
        },
        'training_info': {
            'best_eval_score': training_results['best_eval_score'],
            'final_train_loss': training_results['training_history']['train_loss'][-1] if training_results['training_history']['train_loss'] else None,
            'final_eval_loss': training_results['training_history']['eval_loss'][-1] if training_results['training_history']['eval_loss'] else None
        },
        'finetuned_performance': {
            'exact_match_accuracy': finetuned_results['metrics']['exact_match_accuracy'],
            'valid_json_rate': finetuned_results['metrics']['valid_json_rate'],
            'field_accuracy': finetuned_results['metrics']['field_accuracy']
        }
    }
    
    with open(f"{exp_dir}/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment summary saved to {exp_dir}/experiment_summary.json")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Entity Extraction Fine-tuning')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--entity_loss_weight', type=float, default=2.0, help='Weight for entity loss')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline evaluation')
    parser.add_argument('--data_subset', type=int, help='Use subset of data (for testing)')
    
    args = parser.parse_args()
    
    # Setup experiment
    exp_dir = setup_experiment(args.experiment_name)
    
    # Load and split data
    print("Loading and analyzing data...")
    analyzer = DataAnalyzer('data/event_text_mapping.jsonl')
    analyzer.print_analysis()
    
    train_data, val_data, test_data = analyzer.create_splits()
    
    # Use subset if specified (for testing)
    if args.data_subset:
        train_data = train_data[:args.data_subset]
        val_data = val_data[:args.data_subset//5]
        test_data = test_data[:args.data_subset//5]
        print(f"Using data subset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Setup training configuration
    config = ModelConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        entity_loss_weight=args.entity_loss_weight
    )
    
    # Save config
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump({
            'model_name': config.model_name,
            'max_length': config.max_length,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'entity_loss_weight': config.entity_loss_weight
        }, f, indent=2)
    
    # Run baseline evaluation
    if not args.skip_baseline:
        baseline_results = run_baseline_evaluation(test_data, exp_dir)
    else:
        baseline_results = None
    
    # Run fine-tuning
    training_results = run_fine_tuning(train_data, val_data, config, exp_dir)
    
    # Run final evaluation
    if baseline_results:
        finetuned_results = run_final_evaluation(test_data, exp_dir, baseline_results)
        
        # Save comprehensive summary
        save_experiment_summary(exp_dir, config, baseline_results, training_results, finetuned_results)
    
    # Deployment instructions
    best_model_path = f"{exp_dir}/checkpoints/best_model.pt"
    if os.path.exists(best_model_path):
        print(f"\n{'='*60}")

if __name__ == "__main__":
    main()