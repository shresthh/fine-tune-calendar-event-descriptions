#!/usr/bin/env python3
"""
Run evaluation on existing trained model checkpoint
"""

import os
import torch
import json
from pathlib import Path

from src.core.model_setup import EntityExtractionModel, ModelConfig, load_model
from src.evaluation.baseline_evaluation import BaselineEvaluator, create_evaluation_plots
from src.core.data_analysis import DataAnalyzer

def evaluate_checkpoint(checkpoint_path, test_data, output_prefix="finetuned"):
    """Run evaluation on a specific checkpoint"""
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = ModelConfig()
    
    # Load model and checkpoint weights
    model = load_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model loaded from checkpoint")
    
    # Run evaluation
    evaluator = BaselineEvaluator(model)
    results = evaluator.evaluate_comprehensive(
        test_data, 
        f"{output_prefix}_results.json"
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Create plots
    create_evaluation_plots(results, f"{output_prefix}_plots")
    
    return results

def main():
    """Main evaluation script"""
    
    # Find checkpoint
    checkpoint_candidates = [
        "test_outputs/checkpoint_epoch_0.pt",
        "test_outputs/best_model.pt",
        "experiments/*/checkpoints/checkpoint_epoch_*.pt",
        "experiments/*/checkpoints/best_model.pt"
    ]
    
    checkpoint_path = None
    for pattern in checkpoint_candidates:
        if "*" in pattern:
            from glob import glob
            matches = glob(pattern)
            if matches:
                # Get the most recent one
                checkpoint_path = max(matches, key=lambda f: Path(f).stat().st_mtime)
                break
        elif os.path.exists(pattern):
            checkpoint_path = pattern
            break
    
    if not checkpoint_path:
        print("❌ No checkpoint found!")
        print("Looking for checkpoints in:")
        for pattern in checkpoint_candidates:
            print(f"  - {pattern}")
        
        # List available files
        print("\nAvailable .pt files:")
        for pt_file in Path(".").rglob("*.pt"):
            print(f"  - {pt_file}")
        return
    
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Load test data
    analyzer = DataAnalyzer('data/event_text_mapping.jsonl')
    _, _, test_data = analyzer.create_splits()
    
    # Run evaluation
    try:
        results = evaluate_checkpoint(checkpoint_path, test_data)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED!")
        print("="*60)
        print(f"Results saved:")
        print(f"  - finetuned_results.json: Detailed results")
        print(f"  - finetuned_plots/: Visualization plots")
        
        # Show key metrics
        metrics = results['metrics']
        print(f"\nKey Metrics:")
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
        print(f"  Valid JSON Rate:      {metrics['valid_json_rate']:.3f}")
        print(f"  Complete Fields Rate: {metrics['complete_fields_rate']:.3f}")
        
        # Compare with baseline if available
        if os.path.exists("baseline_results.json"):
            with open("baseline_results.json", 'r') as f:
                baseline = json.load(f)
            
            print(f"\nComparison with Baseline:")
            b_exact = baseline['metrics']['exact_match_accuracy']
            f_exact = metrics['exact_match_accuracy']
            improvement = f_exact - b_exact
            sign = "+" if improvement >= 0 else ""
            print(f"  Exact Match: {b_exact:.3f} → {f_exact:.3f} ({sign}{improvement:.3f})")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()