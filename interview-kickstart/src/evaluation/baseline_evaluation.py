#!/usr/bin/env python3
"""
Baseline Model Evaluation for Entity Extraction
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import re

from src.core.model_setup import EntityExtractionModel, ModelConfig, EntityEvaluator
from src.core.data_analysis import DataAnalyzer, DataPreprocessor
from sklearn.metrics import f1_score, precision_score, recall_score
import re

class BaselineEvaluator:
    """Evaluate baseline model performance on entity extraction"""
    
    def __init__(self, model: EntityExtractionModel):
        self.model = model
        self.evaluator = EntityEvaluator()
        self.entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
    
    def evaluate_comprehensive(self, test_data: List[Dict], 
                              output_file: str = "baseline_results.json") -> Dict[str, Any]:
        """Comprehensive evaluation of baseline model"""
        
        print("Starting comprehensive baseline evaluation...")
        
        results = {
            'total_samples': len(test_data),
            'predictions': [],
            'metrics': {},
            'field_analysis': {},
            'error_analysis': {}
        }
        
        predictions = []
        ground_truths = []
        exact_matches = []
        response_quality = []
        
        # Process each sample
        for i, sample in enumerate(tqdm(test_data, desc="Evaluating baseline")):
            input_text = sample['event_text']
            true_output = sample['output']
            
            # Get model prediction
            try:
                raw_response = self.model.generate_response(input_text)
                predicted_output = self.model.extract_entities(input_text)
                
                # Quality assessment of raw response
                quality = self.assess_response_quality(raw_response, true_output)
                response_quality.append(quality)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                predicted_output = {field: None for field in self.entity_fields}
                raw_response = ""  # Set empty response for error cases
                quality = {
                    'valid_json': False, 
                    'has_all_fields': False, 
                    'response_length': 0,  # Add missing response_length
                    'parse_error': str(e)
                }
                response_quality.append(quality)
            
            # Store results
            result_item = {
                'input': input_text,
                'true_output': true_output,
                'predicted_output': predicted_output,
                'raw_response': raw_response if 'raw_response' in locals() else "",
                'quality': quality
            }
            results['predictions'].append(result_item)
            
            predictions.append(predicted_output)
            ground_truths.append(true_output)
            
            # Calculate exact match
            exact_match = self.evaluator.exact_match_score(predicted_output, true_output)
            exact_matches.append(exact_match)
        
        # Calculate overall metrics
        results['metrics'] = self.calculate_metrics(predictions, ground_truths, exact_matches, response_quality)
        
        # Calculate F1 metrics
        results['f1_metrics'] = self.calculate_f1_metrics(predictions, ground_truths)
        
        # Field-wise analysis
        results['field_analysis'] = self.analyze_field_performance(predictions, ground_truths)
        
        # Error analysis
        results['error_analysis'] = self.analyze_errors(results['predictions'])
        
        # Save results
        with open(output_file, 'w') as f:
            # Make results JSON serializable
            serializable_results = self.make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"Baseline evaluation completed. Results saved to {output_file}")
        
        return results
    
    def assess_response_quality(self, raw_response: str, true_output: Dict) -> Dict[str, Any]:
        """Assess the quality of the model's raw response"""
        quality = {
            'valid_json': False,
            'has_all_fields': False,
            'response_length': len(raw_response),
            'parse_error': None
        }
        
        try:
            # Try to parse as JSON
            parsed = json.loads(raw_response)
            quality['valid_json'] = True
            
            # Check if all expected fields are present
            expected_fields = set(self.entity_fields)
            present_fields = set(parsed.keys())
            quality['has_all_fields'] = expected_fields.issubset(present_fields)
            quality['missing_fields'] = list(expected_fields - present_fields)
            quality['extra_fields'] = list(present_fields - expected_fields)
            
        except json.JSONDecodeError as e:
            quality['parse_error'] = str(e)
            
            # Check if response looks like it's trying to be JSON
            quality['looks_like_json'] = raw_response.strip().startswith('{') and raw_response.strip().endswith('}')
            
            # Check for field mentions
            field_mentions = {}
            for field in self.entity_fields:
                field_mentions[field] = field in raw_response.lower()
            quality['field_mentions'] = field_mentions
        
        return quality
    
    def calculate_metrics(self, predictions: List[Dict], ground_truths: List[Dict], 
                         exact_matches: List[float], response_quality: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        
        metrics = {
            # Core metrics
            'exact_match_accuracy': np.mean(exact_matches),
            'field_accuracy': self.evaluator.field_accuracy(predictions, ground_truths),
            
            # Response quality metrics
            'valid_json_rate': np.mean([q.get('valid_json', False) for q in response_quality]),
            'complete_fields_rate': np.mean([q.get('has_all_fields', False) for q in response_quality]),
            'avg_response_length': np.mean([q.get('response_length', 0) for q in response_quality]),
            
            # Distribution metrics
            'exact_match_distribution': {
                'perfect_match': sum(1 for em in exact_matches if em == 1.0) / len(exact_matches),
                'partial_match': sum(1 for em in exact_matches if 0 < em < 1.0) / len(exact_matches),
                'no_match': sum(1 for em in exact_matches if em == 0.0) / len(exact_matches)
            }
        }
        
        return metrics
    
    def analyze_field_performance(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
        """Analyze performance for each entity field"""
        
        field_analysis = {}
        
        for field in self.entity_fields:
            analysis = {
                'accuracy': 0.0,
                'total_samples': len(predictions),
                'null_in_ground_truth': 0,
                'null_in_predictions': 0,
                'correct_predictions': 0,
                'false_positives': 0,  # Predicted non-null when true is null
                'false_negatives': 0,  # Predicted null when true is non-null
                'common_errors': defaultdict(int)
            }
            
            correct = 0
            for pred, true in zip(predictions, ground_truths):
                pred_val = pred.get(field)
                true_val = true.get(field)
                
                # Count nulls
                if true_val is None:
                    analysis['null_in_ground_truth'] += 1
                if pred_val is None:
                    analysis['null_in_predictions'] += 1
                
                # Check correctness
                if self.values_equal(pred_val, true_val):
                    correct += 1
                    analysis['correct_predictions'] += 1
                else:
                    # Analyze error type
                    if true_val is None and pred_val is not None:
                        analysis['false_positives'] += 1
                    elif true_val is not None and pred_val is None:
                        analysis['false_negatives'] += 1
                    
                    # Record common errors
                    error_key = f"true:{true_val} -> pred:{pred_val}"
                    analysis['common_errors'][error_key] += 1
            
            analysis['accuracy'] = correct / len(predictions)
            
            # Convert defaultdict to regular dict for JSON serialization
            analysis['common_errors'] = dict(analysis['common_errors'])
            
            field_analysis[field] = analysis
        
        return field_analysis
    
    def analyze_errors(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Analyze common error patterns"""
        
        error_analysis = {
            'json_parse_errors': 0,
            'incomplete_responses': 0,
            'common_failure_patterns': defaultdict(int),
            'field_extraction_issues': defaultdict(list)
        }
        
        for i, pred_item in enumerate(predictions):
            quality = pred_item['quality']
            
            # Count parse errors
            if not quality['valid_json']:
                error_analysis['json_parse_errors'] += 1
                
                # Analyze failure patterns
                raw_response = pred_item['raw_response']
                if len(raw_response) == 0:
                    error_analysis['common_failure_patterns']['empty_response'] += 1
                elif not raw_response.strip().startswith('{'):
                    error_analysis['common_failure_patterns']['non_json_format'] += 1
                elif quality.get('parse_error'):
                    error_analysis['common_failure_patterns']['json_syntax_error'] += 1
            
            # Count incomplete responses
            if not quality['has_all_fields']:
                error_analysis['incomplete_responses'] += 1
                
                # Track missing fields
                missing = quality.get('missing_fields', [])
                for field in missing:
                    error_analysis['field_extraction_issues'][field].append(i)
        
        # Convert defaultdicts to regular dicts
        error_analysis['common_failure_patterns'] = dict(error_analysis['common_failure_patterns'])
        error_analysis['field_extraction_issues'] = {
            k: len(v) for k, v in error_analysis['field_extraction_issues'].items()
        }
        
        return error_analysis
    
    def values_equal(self, val1: Any, val2: Any) -> bool:
        """Check if two values are equal, handling lists specially"""
        if isinstance(val1, list) and isinstance(val2, list):
            return set(val1) == set(val2)
        return val1 == val2
    
    def calculate_f1_metrics(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
        """Calculate F1, precision, and recall metrics for each field"""
        
        f1_metrics = {}
        
        for field in self.entity_fields:
            # Extract field values
            pred_values = [pred.get(field) for pred in predictions]
            true_values = [true.get(field) for true in ground_truths]
            
            # Calculate different types of F1 scores
            f1_metrics[field] = {
                'exact_f1': self._calculate_exact_f1(pred_values, true_values),
                'token_f1': self._calculate_token_f1(pred_values, true_values),
                'binary_f1': self._calculate_binary_f1(pred_values, true_values)
            }
        
        # Calculate macro-averaged F1 across all fields
        exact_f1_scores = [metrics['exact_f1']['f1'] for metrics in f1_metrics.values()]
        token_f1_scores = [metrics['token_f1']['f1'] for metrics in f1_metrics.values()]
        binary_f1_scores = [metrics['binary_f1']['f1'] for metrics in f1_metrics.values()]
        
        f1_metrics['macro_averages'] = {
            'exact_f1': np.mean(exact_f1_scores),
            'token_f1': np.mean(token_f1_scores),
            'binary_f1': np.mean(binary_f1_scores)
        }
        
        return f1_metrics
    
    def _calculate_exact_f1(self, pred_values: List[Any], true_values: List[Any]) -> Dict[str, float]:
        """Calculate F1 based on exact field matches"""
        
        # Convert to binary: 1 if exact match, 0 otherwise
        y_true = []
        y_pred = []
        
        for pred, true in zip(pred_values, true_values):
            true_match = 1 if true is not None else 0
            pred_match = 1 if (pred is not None and self.values_equal(pred, true)) else 0
            
            y_true.append(true_match)
            y_pred.append(pred_match)
        
        if sum(y_true) == 0 and sum(y_pred) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except:
            precision = recall = f1 = 0.0
            
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _calculate_token_f1(self, pred_values: List[Any], true_values: List[Any]) -> Dict[str, float]:
        """Calculate F1 based on token-level overlap (useful for text fields)"""
        
        total_precision = 0.0
        total_recall = 0.0
        valid_samples = 0
        
        for pred, true in zip(pred_values, true_values):
            if true is None and pred is None:
                continue
                
            # Convert to token sets
            true_tokens = self._extract_tokens(true)
            pred_tokens = self._extract_tokens(pred)
            
            if len(true_tokens) == 0 and len(pred_tokens) == 0:
                continue
                
            # Calculate token-level precision and recall
            if len(pred_tokens) > 0:
                precision = len(true_tokens & pred_tokens) / len(pred_tokens)
            else:
                precision = 1.0 if len(true_tokens) == 0 else 0.0
                
            if len(true_tokens) > 0:
                recall = len(true_tokens & pred_tokens) / len(true_tokens)
            else:
                recall = 1.0 if len(pred_tokens) == 0 else 0.0
            
            total_precision += precision
            total_recall += recall
            valid_samples += 1
        
        if valid_samples == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
            
        avg_precision = total_precision / valid_samples
        avg_recall = total_recall / valid_samples
        
        if avg_precision + avg_recall > 0:
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1 = 0.0
            
        return {'precision': avg_precision, 'recall': avg_recall, 'f1': f1}
    
    def _calculate_binary_f1(self, pred_values: List[Any], true_values: List[Any]) -> Dict[str, float]:
        """Calculate F1 for presence/absence of field values"""
        
        y_true = [1 if val is not None else 0 for val in true_values]
        y_pred = [1 if val is not None else 0 for val in pred_values]
        
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except:
            precision = recall = f1 = 0.0
            
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _extract_tokens(self, value: Any) -> set:
        """Extract tokens from a field value for token-level F1 calculation"""
        if value is None:
            return set()
            
        if isinstance(value, list):
            # For list fields like attendees
            tokens = set()
            for item in value:
                if item:
                    tokens.update(re.findall(r'\w+', str(item).lower()))
            return tokens
        else:
            # For string fields
            return set(re.findall(r'\w+', str(value).lower()))
    
    def make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of baseline evaluation results"""
        
        print("\n" + "="*60)
        print("BASELINE MODEL EVALUATION SUMMARY")
        print("="*60)
        
        metrics = results['metrics']
        
        print(f"\nOverall Performance:")
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
        print(f"  Valid JSON Rate:      {metrics['valid_json_rate']:.3f}")
        print(f"  Complete Fields Rate: {metrics['complete_fields_rate']:.3f}")
        
        print(f"\nField-wise Accuracy:")
        for field, accuracy in metrics['field_accuracy'].items():
            print(f"  {field:<12}: {accuracy:.3f}")
        
        # Print F1 metrics
        if 'f1_metrics' in results:
            f1_metrics = results['f1_metrics']
            print(f"\nMacro-averaged F1 Scores:")
            print(f"  Exact F1:    {f1_metrics['macro_averages']['exact_f1']:.3f}")
            print(f"  Token F1:    {f1_metrics['macro_averages']['token_f1']:.3f}")
            print(f"  Binary F1:   {f1_metrics['macro_averages']['binary_f1']:.3f}")
            
            print(f"\nField-wise F1 Scores (Exact):")
            for field in self.entity_fields:
                if field in f1_metrics:
                    f1_score = f1_metrics[field]['exact_f1']['f1']
                    precision = f1_metrics[field]['exact_f1']['precision']
                    recall = f1_metrics[field]['exact_f1']['recall']
                    print(f"  {field:<12}: F1={f1_score:.3f} P={precision:.3f} R={recall:.3f}")
        
        print(f"\nMatch Distribution:")
        dist = metrics['exact_match_distribution']
        print(f"  Perfect Match: {dist['perfect_match']:.3f}")
        print(f"  Partial Match: {dist['partial_match']:.3f}")
        print(f"  No Match:      {dist['no_match']:.3f}")
        
        error_analysis = results['error_analysis']
        print(f"\nError Analysis:")
        print(f"  JSON Parse Errors:    {error_analysis['json_parse_errors']}")
        print(f"  Incomplete Responses: {error_analysis['incomplete_responses']}")
        
        # Show top error patterns
        if error_analysis['common_failure_patterns']:
            print(f"\nCommon Failure Patterns:")
            for pattern, count in error_analysis['common_failure_patterns'].items():
                print(f"  {pattern}: {count}")

def create_evaluation_plots(results: Dict[str, Any], output_dir: str = "baseline_plots"):
    """Create visualization plots for baseline evaluation"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Field accuracy plot
    field_accuracy = results['metrics']['field_accuracy']
    
    plt.figure(figsize=(12, 6))
    fields = list(field_accuracy.keys())
    accuracies = list(field_accuracy.values())
    
    bars = plt.bar(fields, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Baseline Model: Field-wise Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Entity Fields', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/field_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Match distribution pie chart
    dist = results['metrics']['exact_match_distribution']
    
    plt.figure(figsize=(8, 8))
    labels = ['Perfect Match', 'Partial Match', 'No Match']
    sizes = [dist['perfect_match'], dist['partial_match'], dist['no_match']]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Baseline Model: Match Distribution', fontsize=16, fontweight='bold')
    plt.savefig(f"{output_dir}/match_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to {output_dir}/")

def main():
    """Run baseline evaluation"""
    
    # Load test data
    analyzer = DataAnalyzer('data/event_text_mapping.jsonl')
    _, _, test_data = analyzer.create_splits()
    
    # Load baseline model
    config = ModelConfig()
    from src.core.model_setup import load_model
    model = load_model(config)
    
    # Run evaluation
    evaluator = BaselineEvaluator(model)
    results = evaluator.evaluate_comprehensive(test_data[:50], "baseline_results.json")  # Sample for demo
    
    # Print summary
    evaluator.print_summary(results)
    
    # Create plots
    create_evaluation_plots(results)
    
    print("\nBaseline evaluation completed!")
    print("Check baseline_results.json for detailed results")
    print("Check baseline_plots/ for visualization")

if __name__ == "__main__":
    main()