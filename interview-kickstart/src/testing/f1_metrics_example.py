#!/usr/bin/env python3
"""
Example demonstrating F1 metrics for entity extraction evaluation
"""

from src.evaluation.baseline_evaluation import BaselineEvaluator
from src.core.model_setup import EntityExtractionModel, ModelConfig
import json

def demonstrate_f1_metrics():
    """Show how F1 metrics would work with sample data"""
    
    # Sample predictions and ground truth
    sample_predictions = [
        {
            "action": "Team meeting",
            "date": "2024-01-15",
            "time": "10:00 AM",
            "attendees": ["Alice", "Bob"],
            "location": "Conference Room A",
            "duration": "1 hour",
            "recurrence": None,
            "notes": None
        },
        {
            "action": "Project review",  # Slightly different from ground truth
            "date": "2024-01-16",
            "time": "2:00 PM",
            "attendees": ["Charlie"],  # Missing one attendee
            "location": "Zoom",
            "duration": "45 minutes",
            "recurrence": None,
            "notes": "Important meeting"
        },
        {
            "action": None,  # Missing action
            "date": "2024-01-17",
            "time": "9:00 AM",
            "attendees": None,
            "location": None,
            "duration": None,
            "recurrence": None,
            "notes": None
        }
    ]
    
    sample_ground_truth = [
        {
            "action": "Team meeting",
            "date": "2024-01-15",
            "time": "10:00 AM",
            "attendees": ["Alice", "Bob"],
            "location": "Conference Room A",
            "duration": "1 hour",
            "recurrence": None,
            "notes": None
        },
        {
            "action": "Project review meeting",  # Longer version
            "date": "2024-01-16",
            "time": "2:00 PM",
            "attendees": ["Charlie", "Dave"],  # Two attendees
            "location": "Zoom",
            "duration": "45 minutes",
            "recurrence": None,
            "notes": "Important meeting"
        },
        {
            "action": "Standup meeting",  # Has action in ground truth
            "date": "2024-01-17",
            "time": "9:00 AM",
            "attendees": ["Team"],
            "location": "Office",
            "duration": "15 minutes",
            "recurrence": "daily",
            "notes": "Daily standup"
        }
    ]
    
    # Create a dummy evaluator (we'll use its F1 calculation methods)
    class DummyModel:
        pass
    
    evaluator = BaselineEvaluator(DummyModel())
    
    # Calculate F1 metrics
    f1_metrics = evaluator.calculate_f1_metrics(sample_predictions, sample_ground_truth)
    
    print("F1 Metrics Analysis Example")
    print("=" * 50)
    
    print(f"\nMacro-averaged F1 Scores:")
    print(f"  Exact F1:    {f1_metrics['macro_averages']['exact_f1']:.3f}")
    print(f"  Token F1:    {f1_metrics['macro_averages']['token_f1']:.3f}")
    print(f"  Binary F1:   {f1_metrics['macro_averages']['binary_f1']:.3f}")
    
    print(f"\nDetailed Field Analysis:")
    for field in evaluator.entity_fields:
        if field in f1_metrics:
            exact = f1_metrics[field]['exact_f1']
            token = f1_metrics[field]['token_f1']
            binary = f1_metrics[field]['binary_f1']
            
            print(f"\n{field.upper()}:")
            print(f"  Exact F1:  P={exact['precision']:.3f} R={exact['recall']:.3f} F1={exact['f1']:.3f}")
            print(f"  Token F1:  P={token['precision']:.3f} R={token['recall']:.3f} F1={token['f1']:.3f}")
            print(f"  Binary F1: P={binary['precision']:.3f} R={binary['recall']:.3f} F1={binary['f1']:.3f}")
    
    print(f"\nWhat these metrics tell us:")
    print(f"- Exact F1: Measures exact field matches (strictest)")
    print(f"- Token F1: Gives partial credit for token overlap (e.g., 'Team meeting' vs 'Team meeting today')")
    print(f"- Binary F1: Measures presence/absence of field values (most lenient)")
    
    return f1_metrics

if __name__ == "__main__":
    demonstrate_f1_metrics()