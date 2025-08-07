#!/usr/bin/env python3
"""
Evaluate fine-tuned model with comprehensive F1 accuracy metrics
"""

import os
import torch
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.model_setup import EntityExtractionModel, ModelConfig, load_model
from src.evaluation.baseline_evaluation import BaselineEvaluator, create_evaluation_plots
from src.core.data_analysis import DataAnalyzer

def create_f1_comparison_plots(results, baseline_results=None, output_dir="f1_plots"):
    """Create comprehensive F1 metric visualization plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if 'f1_metrics' not in results:
        print("⚠️  No F1 metrics found in results")
        return
    
    f1_metrics = results['f1_metrics']
    entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
    
    # 1. Field-wise F1 Score Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fine-tuned Model: F1 Metrics Analysis', fontsize=16, fontweight='bold')
    
    # Exact F1 scores
    exact_f1_scores = [f1_metrics[field]['exact_f1']['f1'] for field in entity_fields if field in f1_metrics]
    exact_precision = [f1_metrics[field]['exact_f1']['precision'] for field in entity_fields if field in f1_metrics]
    exact_recall = [f1_metrics[field]['exact_f1']['recall'] for field in entity_fields if field in f1_metrics]
    
    # Plot 1: Exact F1 Scores
    x = range(len(entity_fields))
    width = 0.25
    
    axes[0, 0].bar([i - width for i in x], exact_precision, width, label='Precision', alpha=0.7, color='skyblue')
    axes[0, 0].bar(x, exact_f1_scores, width, label='F1', alpha=0.7, color='orange')
    axes[0, 0].bar([i + width for i in x], exact_recall, width, label='Recall', alpha=0.7, color='lightgreen')
    
    axes[0, 0].set_title('Exact F1 Metrics by Field')
    axes[0, 0].set_xlabel('Entity Fields')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(entity_fields, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Token F1 Scores
    token_f1_scores = [f1_metrics[field]['token_f1']['f1'] for field in entity_fields if field in f1_metrics]
    token_precision = [f1_metrics[field]['token_f1']['precision'] for field in entity_fields if field in f1_metrics]
    token_recall = [f1_metrics[field]['token_f1']['recall'] for field in entity_fields if field in f1_metrics]
    
    axes[0, 1].bar([i - width for i in x], token_precision, width, label='Precision', alpha=0.7, color='skyblue')
    axes[0, 1].bar(x, token_f1_scores, width, label='F1', alpha=0.7, color='orange')
    axes[0, 1].bar([i + width for i in x], token_recall, width, label='Recall', alpha=0.7, color='lightgreen')
    
    axes[0, 1].set_title('Token F1 Metrics by Field')
    axes[0, 1].set_xlabel('Entity Fields')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(entity_fields, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Binary F1 Scores
    binary_f1_scores = [f1_metrics[field]['binary_f1']['f1'] for field in entity_fields if field in f1_metrics]
    binary_precision = [f1_metrics[field]['binary_f1']['precision'] for field in entity_fields if field in f1_metrics]
    binary_recall = [f1_metrics[field]['binary_f1']['recall'] for field in entity_fields if field in f1_metrics]
    
    axes[1, 0].bar([i - width for i in x], binary_precision, width, label='Precision', alpha=0.7, color='skyblue')
    axes[1, 0].bar(x, binary_f1_scores, width, label='F1', alpha=0.7, color='orange')
    axes[1, 0].bar([i + width for i in x], binary_recall, width, label='Recall', alpha=0.7, color='lightgreen')
    
    axes[1, 0].set_title('Binary F1 Metrics by Field')
    axes[1, 0].set_xlabel('Entity Fields')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(entity_fields, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Macro F1 Comparison
    macro_scores = [
        f1_metrics['macro_averages']['exact_f1'],
        f1_metrics['macro_averages']['token_f1'],
        f1_metrics['macro_averages']['binary_f1']
    ]
    macro_labels = ['Exact F1', 'Token F1', 'Binary F1']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    bars = axes[1, 1].bar(macro_labels, macro_scores, color=colors, alpha=0.7)
    axes[1, 1].set_title('Macro-averaged F1 Scores')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, score in zip(bars, macro_scores):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_metrics_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. F1 vs Accuracy Comparison Plot
    if 'field_accuracy' in results['metrics']:
        field_accuracy = results['metrics']['field_accuracy']
        
        plt.figure(figsize=(12, 8))
        
        # Prepare data for comparison
        fields = [f for f in entity_fields if f in f1_metrics and f in field_accuracy]
        accuracy_scores = [field_accuracy[f] for f in fields]
        exact_f1 = [f1_metrics[f]['exact_f1']['f1'] for f in fields]
        token_f1 = [f1_metrics[f]['token_f1']['f1'] for f in fields]
        
        x = range(len(fields))
        width = 0.25
        
        plt.bar([i - width for i in x], accuracy_scores, width, label='Current Accuracy', alpha=0.8, color='#3498db')
        plt.bar(x, exact_f1, width, label='Exact F1', alpha=0.8, color='#e74c3c')
        plt.bar([i + width for i in x], token_f1, width, label='Token F1', alpha=0.8, color='#2ecc71')
        
        plt.title('Accuracy vs F1 Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Entity Fields')
        plt.ylabel('Score')
        plt.xticks(x, fields, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_vs_f1_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"F1 analysis plots saved to {output_dir}/")

def print_detailed_f1_analysis(results):
    """Print comprehensive F1 analysis"""
    
    if 'f1_metrics' not in results:
        print("⚠️  No F1 metrics available")
        return
    
    f1_metrics = results['f1_metrics']
    entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE F1 METRICS ANALYSIS")
    print("="*80)
    
    # Macro averages
    print(f"\n📊 MACRO-AVERAGED F1 SCORES:")
    print(f"  Exact F1:    {f1_metrics['macro_averages']['exact_f1']:.3f}")
    print(f"  Token F1:    {f1_metrics['macro_averages']['token_f1']:.3f}")
    print(f"  Binary F1:   {f1_metrics['macro_averages']['binary_f1']:.3f}")
    
    # Field-wise detailed analysis
    print(f"\n📋 DETAILED FIELD-WISE F1 ANALYSIS:")
    print(f"{'Field':<12} {'Exact F1':<10} {'Token F1':<10} {'Binary F1':<10} {'Best Metric'}")
    print("-" * 60)
    
    for field in entity_fields:
        if field in f1_metrics:
            exact_f1 = f1_metrics[field]['exact_f1']['f1']
            token_f1 = f1_metrics[field]['token_f1']['f1']
            binary_f1 = f1_metrics[field]['binary_f1']['f1']
            
            # Determine best metric
            best_score = max(exact_f1, token_f1, binary_f1)
            if best_score == exact_f1:
                best_metric = "Exact"
            elif best_score == token_f1:
                best_metric = "Token"
            else:
                best_metric = "Binary"
            
            print(f"{field:<12} {exact_f1:<10.3f} {token_f1:<10.3f} {binary_f1:<10.3f} {best_metric} ({best_score:.3f})")
    
    # Precision/Recall breakdown for problematic fields
    print(f"\n🔍 PRECISION/RECALL BREAKDOWN FOR LOW-PERFORMING FIELDS:")
    
    low_performing_fields = []
    for field in entity_fields:
        if field in f1_metrics:
            exact_f1 = f1_metrics[field]['exact_f1']['f1']
            if exact_f1 < 0.5:  # Consider fields with F1 < 0.5 as low-performing
                low_performing_fields.append(field)
    
    if low_performing_fields:
        for field in low_performing_fields:
            exact_metrics = f1_metrics[field]['exact_f1']
            print(f"\n  {field.upper()}:")
            print(f"    Precision: {exact_metrics['precision']:.3f}")
            print(f"    Recall:    {exact_metrics['recall']:.3f}")
            print(f"    F1 Score:  {exact_metrics['f1']:.3f}")
            
            # Interpretation
            if exact_metrics['precision'] > exact_metrics['recall']:
                print(f"    → Model is conservative (high precision, low recall)")
            elif exact_metrics['recall'] > exact_metrics['precision']:
                print(f"    → Model is liberal (low precision, high recall)")
            else:
                print(f"    → Balanced precision and recall")
    else:
        print("  All fields performing well (F1 >= 0.5)!")

def evaluate_finetuned_with_f1(checkpoint_path, test_data, output_prefix="finetuned_f1"):
    """Run comprehensive F1 evaluation on fine-tuned model"""
    
    print(f"🚀 Evaluating fine-tuned model with F1 metrics: {checkpoint_path}")
    
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
    print("✅ Fine-tuned model loaded successfully")
    
    # Run comprehensive evaluation with F1 metrics
    evaluator = BaselineEvaluator(model)
    results = evaluator.evaluate_comprehensive(
        test_data, 
        f"{output_prefix}_results.json"
    )
    
    # Print standard summary
    evaluator.print_summary(results)
    
    # Print detailed F1 analysis
    print_detailed_f1_analysis(results)
    
    # Create F1-specific plots
    create_f1_comparison_plots(results, output_dir=f"{output_prefix}_plots")
    
    # Create standard plots too
    create_evaluation_plots(results, f"{output_prefix}_standard_plots")
    
    return results

def main():
    """Main evaluation script with F1 focus"""
    
    print("🎯 Fine-tuned Model F1 Evaluation")
    print("="*50)
    
    # Find the best checkpoint
    checkpoint_candidates = [
        "experiments/entity_extraction_20250803_001100/checkpoints/best_model.pt",
        "test_outputs/best_model.pt",
        "experiments/*/checkpoints/best_model.pt",
        "test_outputs/checkpoint_epoch_*.pt"
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
        print("❌ No fine-tuned checkpoint found!")
        print("Looking for checkpoints in:")
        for pattern in checkpoint_candidates:
            print(f"  - {pattern}")
        return
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    
    # Load test data
    print("📂 Loading test dataset...")
    analyzer = DataAnalyzer('data/event_text_mapping.jsonl')
    _, _, test_data = analyzer.create_splits()
    print(f"✅ Loaded {len(test_data)} test samples")
    
    # Run F1 evaluation
    try:
        results = evaluate_finetuned_with_f1(checkpoint_path, test_data)
        
        print("\n" + "="*80)
        print("🎉 F1 EVALUATION COMPLETED!")
        print("="*80)
        print(f"📁 Results saved:")
        print(f"  - finetuned_f1_results.json: Comprehensive results with F1 metrics")
        print(f"  - finetuned_f1_plots/: F1-specific visualization plots")
        print(f"  - finetuned_f1_standard_plots/: Standard evaluation plots")
        
        # Show summary of key F1 metrics
        if 'f1_metrics' in results:
            f1_metrics = results['f1_metrics']
            print(f"\n🏆 KEY F1 PERFORMANCE SUMMARY:")
            print(f"  Overall Exact Match F1:  {f1_metrics['macro_averages']['exact_f1']:.3f}")
            print(f"  Overall Token F1:        {f1_metrics['macro_averages']['token_f1']:.3f}")
            print(f"  Overall Binary F1:       {f1_metrics['macro_averages']['binary_f1']:.3f}")
            
            # Highlight best and worst performing fields
            entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
            field_f1_scores = [(field, f1_metrics[field]['exact_f1']['f1']) for field in entity_fields if field in f1_metrics]
            field_f1_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n  🥇 Best performing field:  {field_f1_scores[0][0]} (F1: {field_f1_scores[0][1]:.3f})")
            print(f"  🥉 Worst performing field: {field_f1_scores[-1][0]} (F1: {field_f1_scores[-1][1]:.3f})")
        
    except Exception as e:
        print(f"❌ F1 evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()