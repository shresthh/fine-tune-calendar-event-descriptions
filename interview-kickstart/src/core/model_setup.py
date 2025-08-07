#!/usr/bin/env python3
"""
Model Setup and Custom Loss Function for Entity Extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the model and training"""
    model_name: str = "HuggingFaceTB/SmolLM-360M"
    max_length: int = 512
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    eval_steps: int = 100
    save_steps: int = 500
    entity_loss_weight: float = 2.0  # Higher weight for entity-specific loss

class EntityExtractionModel(nn.Module):
    """Wrapper around SmolLM for entity extraction with custom loss"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Entity fields for structured loss
        self.entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
        
        # Special tokens for entity extraction
        self.separator_token = " → "
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with custom loss calculation"""
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if labels is not None else None
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # If training, compute custom entity-aware loss
        if labels is not None and self.training:
            custom_loss = self.compute_entity_loss(input_ids, logits, labels, attention_mask)
            total_loss = loss + self.config.entity_loss_weight * custom_loss
        else:
            total_loss = loss
            custom_loss = torch.tensor(0.0, device=loss.device if loss is not None else 'cpu')
        
        return {
            'loss': total_loss,
            'base_loss': loss,
            'entity_loss': custom_loss,
            'logits': logits
        }
    
    def compute_entity_loss(self, input_ids: torch.Tensor, logits: torch.Tensor, 
                           labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute custom loss that focuses on entity extraction accuracy"""
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        total_entity_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for i in range(batch_size):
            # Find separator token to identify output portion
            input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            
            if self.separator_token in input_text:
                # Find the position of the separator
                separator_tokens = self.tokenizer.encode(self.separator_token, add_special_tokens=False)
                
                # Find separator position in token sequence
                sep_pos = None
                for j in range(len(input_ids[i]) - len(separator_tokens) + 1):
                    if torch.equal(input_ids[i][j:j+len(separator_tokens)], 
                                 torch.tensor(separator_tokens, device=device)):
                        sep_pos = j + len(separator_tokens)
                        break
                
                if sep_pos is not None and sep_pos < len(input_ids[i]):
                    # Focus loss on the output portion (after separator)
                    output_logits = logits[i][sep_pos-1:-1]  # Shift for causal LM
                    output_labels = labels[i][sep_pos:]
                    output_mask = attention_mask[i][sep_pos:]
                    
                    # Compute cross-entropy loss for output portion
                    if len(output_logits) > 0 and len(output_labels) > 0:
                        min_len = min(len(output_logits), len(output_labels))
                        output_logits = output_logits[:min_len]
                        output_labels = output_labels[:min_len]
                        output_mask = output_mask[:min_len]
                        
                        # Only compute loss on non-padded tokens
                        if output_mask.sum() > 0:
                            ce_loss = F.cross_entropy(
                                output_logits.view(-1, output_logits.size(-1)),
                                output_labels.view(-1),
                                reduction='none'
                            )
                            
                            # Apply mask and average
                            masked_loss = (ce_loss.view(-1) * output_mask.float()).sum() / output_mask.sum()
                            total_entity_loss += masked_loss
                            valid_samples += 1
        
        return total_entity_loss / max(valid_samples, 1)
    
    def generate_response(self, input_text: str, max_new_tokens: int = 200, 
                         temperature: float = 0.1, do_sample: bool = True) -> str:
        """Generate entity extraction response"""
        
        # Format input
        formatted_input = f"Extract entities from: {input_text}"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors='pt',
            max_length=self.config.max_length - max_new_tokens,
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if self.separator_token in full_response:
            response = full_response.split(self.separator_token, 1)[1].strip()
        else:
            response = full_response[len(formatted_input):].strip()
        
        return response
    
    def extract_entities(self, input_text: str) -> Dict[str, Any]:
        """Extract entities and return as structured dictionary"""
        
        response = self.generate_response(input_text)
        
        try:
            # Try to parse as JSON
            entities = json.loads(response)
            
            # Ensure entities is a dictionary
            if not isinstance(entities, dict):
                # If parsed JSON is not a dict (e.g., number, string, list), fallback
                return {field: None for field in self.entity_fields}
            
            # Ensure all expected fields are present
            for field in self.entity_fields:
                if field not in entities:
                    entities[field] = None
            
            return entities
            
        except json.JSONDecodeError:
            # Fallback: return None for all fields if parsing fails
            return {field: None for field in self.entity_fields}

class EntityEvaluator:
    """Evaluate entity extraction performance"""
    
    def __init__(self):
        self.entity_fields = ['action', 'date', 'time', 'attendees', 'location', 'duration', 'recurrence', 'notes']
    
    def exact_match_score(self, predicted: Dict, ground_truth: Dict) -> float:
        """Calculate exact match score for all entities"""
        matches = 0
        total = len(self.entity_fields)
        
        for field in self.entity_fields:
            pred_val = predicted.get(field)
            true_val = ground_truth.get(field)
            
            # Handle list comparison for attendees
            if isinstance(pred_val, list) and isinstance(true_val, list):
                if set(pred_val) == set(true_val):
                    matches += 1
            elif pred_val == true_val:
                matches += 1
        
        return matches / total
    
    def field_accuracy(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
        """Calculate per-field accuracy"""
        field_scores = {field: [] for field in self.entity_fields}
        
        for pred, true in zip(predictions, ground_truths):
            for field in self.entity_fields:
                pred_val = pred.get(field)
                true_val = true.get(field)
                
                if isinstance(pred_val, list) and isinstance(true_val, list):
                    score = 1.0 if set(pred_val) == set(true_val) else 0.0
                else:
                    score = 1.0 if pred_val == true_val else 0.0
                
                field_scores[field].append(score)
        
        return {field: np.mean(scores) for field, scores in field_scores.items()}
    
    def evaluate_batch(self, model: EntityExtractionModel, samples: List[Dict]) -> Dict[str, float]:
        """Evaluate model on a batch of samples"""
        predictions = []
        ground_truths = []
        exact_matches = []
        
        for sample in samples:
            # Get prediction
            pred = model.extract_entities(sample['original_input'])
            true = sample['original_output']
            
            predictions.append(pred)
            ground_truths.append(true)
            
            # Calculate exact match
            exact_match = self.exact_match_score(pred, true)
            exact_matches.append(exact_match)
        
        # Calculate metrics
        metrics = {
            'exact_match': np.mean(exact_matches),
            **self.field_accuracy(predictions, ground_truths)
        }
        
        return metrics

def load_model(config: ModelConfig, checkpoint_path: Optional[str] = None) -> EntityExtractionModel:
    """Load model with optional checkpoint"""
    model = EntityExtractionModel(config)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model

def main():
    """Test model setup"""
    config = ModelConfig()
    
    print("Setting up model...")
    model = load_model(config)
    
    print(f"Model loaded: {config.model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test generation
    test_input = "Meeting with John tomorrow at 2pm for 1 hour"
    print(f"\nTest input: {test_input}")
    
    try:
        response = model.generate_response(test_input)
        print(f"Generated response: {response}")
        
        entities = model.extract_entities(test_input)
        print(f"Extracted entities: {json.dumps(entities, indent=2)}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
    
    print("\nModel setup completed successfully!")

if __name__ == "__main__":
    main()