#!/usr/bin/env python3
"""
Data Analysis and Preprocessing for Calendar Event Entity Extraction
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

class DataAnalyzer:
    """Analyze and preprocess the calendar event dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self.load_data()
        self.entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
        
    def load_data(self) -> List[Dict]:
        """Load JSONL data"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        print(f"Loaded {len(data)} samples")
        return data
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Comprehensive dataset analysis"""
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(self.data)
        
        # Entity field presence analysis
        entity_presence = defaultdict(int)
        entity_non_null = defaultdict(int)
        
        for sample in self.data:
            output = sample['output']
            for field in self.entity_fields:
                if field in output:
                    entity_presence[field] += 1
                    if output[field] is not None:
                        entity_non_null[field] += 1
        
        analysis['entity_presence'] = dict(entity_presence)
        analysis['entity_non_null'] = dict(entity_non_null)
        analysis['entity_null_percentage'] = {
            field: (entity_presence[field] - entity_non_null[field]) / entity_presence[field] * 100
            for field in self.entity_fields
        }
        
        # Text length analysis
        text_lengths = [len(sample['event_text']) for sample in self.data]
        analysis['text_length_stats'] = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths),
            'median': np.median(text_lengths)
        }
        
        # Unique values analysis
        unique_values = defaultdict(set)
        for sample in self.data:
            output = sample['output']
            for field in self.entity_fields:
                if output.get(field) is not None:
                    if isinstance(output[field], list):
                        unique_values[field].update(output[field])
                    else:
                        unique_values[field].add(output[field])
        
        analysis['unique_values_count'] = {field: len(values) for field, values in unique_values.items()}
        
        return analysis
    
    def print_analysis(self):
        """Print comprehensive dataset analysis"""
        analysis = self.analyze_dataset()
        
        print("=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        
        print(f"\nTotal samples: {analysis['total_samples']}")
        
        print("\nEntity Field Presence:")
        for field in self.entity_fields:
            presence = analysis['entity_presence'].get(field, 0)
            non_null = analysis['entity_non_null'].get(field, 0)
            null_pct = analysis['entity_null_percentage'].get(field, 0)
            print(f"  {field:<12}: {non_null:>3}/{presence:>3} non-null ({null_pct:>5.1f}% null)")
        
        print(f"\nText Length Statistics:")
        stats = analysis['text_length_stats']
        print(f"  Mean: {stats['mean']:.1f} chars")
        print(f"  Std:  {stats['std']:.1f} chars")
        print(f"  Min:  {stats['min']} chars")
        print(f"  Max:  {stats['max']} chars")
        print(f"  Median: {stats['median']:.1f} chars")
        
        print(f"\nUnique Values Count:")
        for field in self.entity_fields:
            count = analysis['unique_values_count'].get(field, 0)
            print(f"  {field:<12}: {count:>3} unique values")
    
    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, random_state: int = 42) -> Tuple[List, List, List]:
        """Create train/validation/test splits"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # First split into train and temp (val + test)
        train_data, temp_data = train_test_split(
            self.data, test_size=(val_ratio + test_ratio), random_state=random_state
        )
        
        # Then split temp into val and test
        val_data, test_data = train_test_split(
            temp_data, test_size=test_ratio/(val_ratio + test_ratio), random_state=random_state
        )
        
        print(f"Dataset splits:")
        print(f"  Train: {len(train_data)} samples ({len(train_data)/len(self.data)*100:.1f}%)")
        print(f"  Val:   {len(val_data)} samples ({len(val_data)/len(self.data)*100:.1f}%)")
        print(f"  Test:  {len(test_data)} samples ({len(test_data)/len(self.data)*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def show_sample_examples(self, n_samples: int = 5):
        """Show sample examples from the dataset"""
        print("\n" + "=" * 60)
        print("SAMPLE EXAMPLES")
        print("=" * 60)
        
        for i, sample in enumerate(self.data[:n_samples]):
            print(f"\nExample {i+1}:")
            print(f"Input:  {sample['event_text']}")
            print(f"Output: {json.dumps(sample['output'], indent=2)}")

class DataPreprocessor:
    """Preprocess data for model training"""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM-360M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
    
    def format_input_output(self, event_text: str, output_dict: Dict) -> Tuple[str, str]:
        """Format input and expected output for training"""
        # Input format: "Extract entities from: {event_text}"
        input_text = f"Extract entities from: {event_text}"
        
        # Output format: JSON string
        output_text = json.dumps(output_dict, separators=(',', ':'))
        
        return input_text, output_text
    
    def tokenize_sample(self, input_text: str, output_text: str, max_length: int = 512) -> Dict:
        """Tokenize a single sample for training"""
        # Combine input and output for language modeling
        full_text = f"{input_text} → {output_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids (shifted internally by the model)
        encoding['labels'] = encoding['input_ids'].clone()
        
        return {k: v.squeeze(0) for k, v in encoding.items()}
    
    def prepare_dataset(self, data: List[Dict], max_length: int = 512) -> List[Dict]:
        """Prepare entire dataset for training"""
        processed_data = []
        
        for sample in data:
            input_text, output_text = self.format_input_output(
                sample['event_text'], sample['output']
            )
            
            tokenized = self.tokenize_sample(input_text, output_text, max_length)
            
            # Store original data for evaluation
            tokenized['original_input'] = sample['event_text']
            tokenized['original_output'] = sample['output']
            tokenized['formatted_input'] = input_text
            tokenized['formatted_output'] = output_text
            
            processed_data.append(tokenized)
        
        return processed_data

def main():
    """Main analysis and preprocessing"""
    # Initialize analyzer
    analyzer = DataAnalyzer('data/event_text_mapping.jsonl')
    
    # Run analysis
    analyzer.print_analysis()
    analyzer.show_sample_examples()
    
    # Create splits
    train_data, val_data, test_data = analyzer.create_splits()
    
    # Save splits
    with open('data/train_split.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data/val_split.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open('data/test_split.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nSaved data splits to data/ directory")
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    train_processed = preprocessor.prepare_dataset(train_data[:5])
    
    print(f"\nPreprocessing test:")
    print(f"Processed {len(train_processed)} samples")
    print(f"Sample tokenized shape: {train_processed[0]['input_ids'].shape}")

if __name__ == "__main__":
    main()