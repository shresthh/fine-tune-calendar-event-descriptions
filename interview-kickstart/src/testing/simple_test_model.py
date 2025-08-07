#!/usr/bin/env python3
"""
Simple test script for the fine-tuned entity extraction model
"""

import torch
import json
from src.core.model_setup import EntityExtractionModel, ModelConfig, load_model

def test_model(model_path, test_text):
    """Test the fine-tuned model with sample text"""
    
    print("Loading model configuration...")
    config = ModelConfig()
    
    print("Loading base model...")
    model = load_model(config)
    
    print(f"Loading fine-tuned weights from: {model_path}")
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Test the model
    prompt = f"Extract entities from: {test_text}"
    print(f"\nInput: {prompt}")
    
    # Tokenize input
    inputs = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length)
    
    # Generate response
    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=model.tokenizer.eos_token_id
        )
    
    # Decode response
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part
    generated_text = response[len(prompt):].strip()
    print(f"Output: {generated_text}")
    
    # Try to parse as JSON
    try:
        # Look for JSON in the response
        start_idx = generated_text.find('{')
        end_idx = generated_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = generated_text[start_idx:end_idx]
            entities = json.loads(json_str)
            print(f"\nParsed entities:")
            for key, value in entities.items():
                print(f"  {key}: {value}")
        else:
            print("\nNo valid JSON found in response")
    except Exception as e:
        print(f"\nCould not parse JSON: {e}")

def main():
    model_path = "test_outputs/best_model.pt"
    
    # Test examples
    test_examples = [
        "Meeting with John tomorrow at 2pm for 1 hour at the office",
        "Lunch with Sarah on Friday at 12:30pm at the restaurant",
        "Team standup every Monday at 9am for 30 minutes",
        "Birthday party on December 25th at 7pm"
    ]
    
    print("🎯 Testing Fine-tuned Entity Extraction Model")
    print("=" * 50)
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n📝 Test {i}:")
        print("-" * 30)
        try:
            test_model(model_path, example)
        except Exception as e:
            print(f"Error testing model: {e}")
        print("-" * 30)

if __name__ == "__main__":
    main()