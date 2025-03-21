# create alapaca dataset as input
# create aae dataset as output

import json
import os
from together import Together
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Initialize Together.ai API
client = Together()

# Load the Alpaca dataset
alpaca_data = json.load(open('data/alpaca_data.json'))
print(f"Loaded {len(alpaca_data)} examples from Alpaca dataset")

@dataclass
class Translation:
    text: str
    model: str
    field: str  # 'instruction' or 'input'

@dataclass
class ExampleTranslations:
    original: Dict
    translations: List[Translation]
    example_id: int

def convert_to_aae(text: str, model: str) -> str:
    """Convert SAE text to AAE using a specified model."""
    prompt = f"""Convert this SAE text to AAE:

SAE: {text}

AAE:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1,
        )
        text = response.choices[0].message.content.strip()
        
        # Parse the response to get just the translation
        lines = text.split('\n')
        for line in lines:
            if line.startswith('AAE:'):
                return line[4:].strip()
            elif not line.startswith('SAE:'):
                return line.strip()
        
        return text.strip()
    except Exception as e:
        print(f"Error in convert_to_aae with model {model}: {str(e)}")
        return None

def translate_example(example: Dict, example_id: int) -> ExampleTranslations:
    """Translate both instruction and input to AAE using both models."""
    # Models to use for translation
    models = [
        "Qwen/Qwen2-VL-72B-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    ]
    
    translations = []
    
    # Translate instruction
    for model in models:
        translation = convert_to_aae(example['instruction'], model)
        if translation:
            translations.append(Translation(
                text=translation,
                model=model,
                field='instruction'
            ))
    
    # Translate input if it exists
    if example.get('input'):
        for model in models:
            translation = convert_to_aae(example['input'], model)
            if translation:
                translations.append(Translation(
                    text=translation,
                    model=model,
                    field='input'
                ))
    
    return ExampleTranslations(
        original=example,
        translations=translations,
        example_id=example_id
    )

def save_translations(translations: List[ExampleTranslations], output_file: str):
    """Save translations in a format suitable for later scoring."""
    # Convert dataclasses to dictionaries
    data = []
    for trans in translations:
        example_data = {
            'example_id': trans.example_id,
            'original': trans.original,
            'translations': [asdict(t) for t in trans.translations]
        }
        data.append(example_data)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Process only first 5 examples for testing
test_size = 5
alpaca_data = alpaca_data[:test_size]

# Generate translations
all_translations = []
for i, example in enumerate(alpaca_data):
    print(f"\nProcessing example {i+1}/{len(alpaca_data)}")
    
    # Translate example
    translations = translate_example(example, i)
    all_translations.append(translations)
    
    print(f"Generated {len(translations.translations)} translations for example {i+1}")

# Save all translations
output_file = 'data/translation_candidates.json'
save_translations(all_translations, output_file)

print(f"\nSuccessfully processed {len(all_translations)} examples")
print(f"Translation candidates saved to {output_file}")
print("\nNext steps:")
print("1. Run the scoring script to evaluate translations")
print("2. Use the scoring results to select the best translations")
print("3. Generate the final SAE and AAE datasets")

