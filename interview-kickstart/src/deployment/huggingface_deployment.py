#!/usr/bin/env python3
"""
Deploy Fine-tuned Entity Extraction Model to Hugging Face Hub
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, Repository, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile

from src.core.model_setup import EntityExtractionModel, ModelConfig, load_model

class HuggingFaceDeployer:
    """Deploy fine-tuned model to Hugging Face Hub"""
    
    def __init__(self, model_path: str, repo_name: str, hf_token: Optional[str] = None):
        self.model_path = model_path
        self.repo_name = repo_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.config = ModelConfig()
        
        if not self.hf_token:
            raise ValueError("Please provide Hugging Face token via --hf_token or HF_TOKEN environment variable")
    
    def prepare_model_files(self, output_dir: str):
        """Prepare all model files for upload"""
        
        print("📦 Preparing model files...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the fine-tuned model
        print("🔄 Loading fine-tuned model...")
        model = load_model(self.config)
        
        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(self.model_path, weights_only=False)
        else:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save the model and tokenizer in HuggingFace format
        print("💾 Saving model in HuggingFace format...")
        model.model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        
        # Create model card (README.md)
        self.create_model_card(output_dir)
        
        # Create configuration files
        self.create_config_files(output_dir, checkpoint)
        
        print(f"✅ Model files prepared in: {output_dir}")
    
    def create_model_card(self, output_dir: str):
        """Create a comprehensive model card"""
        
        model_card = f"""---
license: apache-2.0
base_model: {self.config.model_name}
tags:
- text-generation
- entity-extraction
- calendar-events
- fine-tuned
- pytorch
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# Entity Extraction Model - Fine-tuned SmolLM-360M

This model is a fine-tuned version of [{self.config.model_name}](https://huggingface.co/{self.config.model_name}) for extracting structured entities from natural language calendar event descriptions.

## Model Description

- **Base Model**: {self.config.model_name}
- **Task**: Entity Extraction for Calendar Events
- **Language**: English
- **License**: Apache 2.0

## Intended Use

This model extracts structured entities from natural language text describing calendar events. It outputs JSON with the following fields:

- `action`: Type of event (e.g., "meeting", "lunch")
- `date`: Date in DD/MM/YYYY format
- `time`: Time in HH:MM AM/PM format
- `attendees`: Array of attendee names (or null)
- `location`: Event location (or null)
- `duration`: Duration description (or null)
- `recurrence`: Recurrence pattern (or null)
- `notes`: Additional notes (or null)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "{self.repo_name}"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example usage
text = "Meeting with John tomorrow at 2pm for 1 hour at the office"
prompt = f"Extract entities from: {{text}}"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_text = response[len(prompt):].strip()
print(generated_text)
```

## Expected Output Format

```json
{{
  "action": "Meeting",
  "date": "tomorrow", 
  "time": "2:00 PM",
  "attendees": ["John"],
  "location": "office",
  "duration": "1 hour",
  "recurrence": null,
  "notes": null
}}
```

## Training Details

- **Training Data**: 793 calendar event samples
- **Training Split**: 70% train, 15% validation, 15% test
- **Custom Loss Function**: Entity-aware loss with weighted output portion
- **Training Framework**: PyTorch (custom trainer)
- **Evaluation Metrics**: Exact match accuracy, field-wise accuracy, JSON quality

## Model Performance

The model demonstrates strong performance in:
- Accurate entity extraction from natural language
- Consistent JSON output format
- Handling of missing/null values
- Recognition of temporal expressions
- Identification of people and locations

## Limitations

- Primarily trained on English calendar events
- May struggle with very complex or ambiguous temporal expressions
- Performance may vary with domain-specific terminology
- Requires specific input format: "Extract entities from: [text]"

## Training Procedure

This model was fine-tuned using:
1. Custom PyTorch trainer implementation
2. Entity-weighted loss function (weight: 2.0)
3. Cosine annealing learning rate schedule
4. Gradient accumulation for effective larger batch sizes
5. Early stopping based on validation performance

## Citation

If you use this model, please cite:

```bibtex
@misc{{entity-extraction-smollm,
  title={{Entity Extraction Fine-tuned SmolLM-360M}},
  author={{Shresth Mishra}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{self.repo_name}}}}}
}}
```

## Contact

For questions about this model, please open an issue in the repository or contact the author.
"""
        
        with open(os.path.join(output_dir, "README.md"), 'w') as f:
            f.write(model_card)
        
        print("📄 Model card created")
    
    def create_config_files(self, output_dir: str, checkpoint: dict):
        """Create additional configuration files"""
        
        # Training configuration
        training_config = {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "entity_loss_weight": self.config.entity_loss_weight,
            "training_completed": True,
            "checkpoint_info": {
                "epoch": checkpoint.get('epoch', 'unknown'),
                "global_step": checkpoint.get('global_step', 'unknown'),
                "best_eval_score": checkpoint.get('best_eval_score', 'unknown')
            }
        }
        
        with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
            json.dump(training_config, f, indent=2)
        
        # Usage example
        usage_example = {
            "example_usage": {
                "input": "Extract entities from: Meeting with John tomorrow at 2pm for 1 hour",
                "expected_output": {
                    "action": "Meeting",
                    "date": "tomorrow",
                    "time": "2:00 PM", 
                    "attendees": ["John"],
                    "location": None,
                    "duration": "1 hour",
                    "recurrence": None,
                    "notes": None
                }
            },
            "supported_fields": [
                "action", "date", "time", "attendees", 
                "location", "duration", "recurrence", "notes"
            ],
            "input_format": "Extract entities from: [your event description]"
        }
        
        with open(os.path.join(output_dir, "usage_example.json"), 'w') as f:
            json.dump(usage_example, f, indent=2)
        
        print("⚙️ Configuration files created")
    
    def deploy_to_hub(self, model_dir: str, private: bool = False):
        """Deploy model to Hugging Face Hub"""
        
        print(f"🚀 Deploying to Hugging Face Hub: {self.repo_name}")
        
        try:
            # Initialize HF API
            api = HfApi(token=self.hf_token)
            
            # Create repository
            print("📝 Creating repository...")
            try:
                create_repo(
                    repo_id=self.repo_name,
                    token=self.hf_token,
                    private=private,
                    exist_ok=True
                )
                print(f"✅ Repository created/exists: https://huggingface.co/{self.repo_name}")
            except Exception as e:
                print(f"⚠️ Repository creation warning: {e}")
            
            # Upload all files
            print("📤 Uploading model files...")
            api.upload_folder(
                folder_path=model_dir,
                repo_id=self.repo_name,
                token=self.hf_token,
                commit_message="Upload fine-tuned entity extraction model"
            )
            
            return {
                "success": True,
                "repo_url": f"https://huggingface.co/{self.repo_name}",
                "message": f"Model successfully deployed to {self.repo_name}!"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Deployment failed: {str(e)}"
            }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy fine-tuned model to Hugging Face Hub')
    parser.add_argument('--model_path', type=str, default='test_outputs/best_model.pt',
                       help='Path to the fine-tuned model checkpoint')
    parser.add_argument('--repo_name', type=str, required=True,
                       help='Repository name on Hugging Face (format: username/model-name)')
    parser.add_argument('--hf_token', type=str,
                       help='Hugging Face token (or set HF_TOKEN env var)')
    parser.add_argument('--private', action='store_true',
                       help='Make repository private')
    parser.add_argument('--output_dir', type=str,
                       help='Local directory to prepare files (temporary if not specified)')
    
    args = parser.parse_args()
    
    print("🤗 Hugging Face Model Deployment")
    print("=" * 50)
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    try:
        deployer = HuggingFaceDeployer(
            model_path=args.model_path,
            repo_name=args.repo_name,
            hf_token=args.hf_token
        )
        
        # Prepare model files
        if args.output_dir:
            model_dir = args.output_dir
            os.makedirs(model_dir, exist_ok=True)
        else:
            model_dir = tempfile.mkdtemp(prefix="hf_model_")
        
        deployer.prepare_model_files(model_dir)
        
        # Deploy to Hub
        result = deployer.deploy_to_hub(model_dir, private=args.private)
        
        if result["success"]:
            print(f"\n🎉 SUCCESS!")
            print(f"📍 Model URL: {result['repo_url']}")
            print(f"\n📋 Usage:")
            print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
            print(f"model = AutoModelForCausalLM.from_pretrained('{args.repo_name}')")
            print(f"tokenizer = AutoTokenizer.from_pretrained('{args.repo_name}')")
            print(f"\n🔗 Share your model: {result['repo_url']}")
        else:
            print(f"\n❌ FAILED: {result['error']}")
        
        # Clean up temporary directory if created
        if not args.output_dir:
            shutil.rmtree(model_dir)
            print(f"🧹 Cleaned up temporary files")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()