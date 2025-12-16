import argparse
import json
import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import requests

# Model configurations - Gemini only
MODEL_CONFIGS = {
    "gemini": {
        "gemini-3-pro-preview": "google/gemini-3-pro-preview", 
        "gemini-2.5-flash": "google/gemini-2.5-flash"
    }
}

# OpenRouter configuration
OPENROUTER_URL = "https://llmfoundry.straive.com/openrouter/v1/chat/completions"

class FeatureSchema:
    """Loads and validates features from features.json"""
    
    def __init__(self, features_path: Path):
        self.features_path = features_path
        self.features = self._load_features()
        self.voice_features = [f for f in self.features if f.get('category') == 'voice']
        self.music_features = [f for f in self.features if f.get('category') == 'music']
    
    def _load_features(self) -> List[Dict]:
        """Load features from JSON file"""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        with open(self.features_path, 'r') as f:
            data = json.load(f)
        
        return data.get('features', [])
    
    def generate_feature_descriptions(self) -> str:
        """Generate feature descriptions for prompt"""
        descriptions = []
        
        for feature in self.features:
            name = feature['name']
            desc = feature['description']
            feature_type = feature['type']
            allowed = feature.get('allowed_values', [])
            
            if allowed:
                allowed_str = ' | '.join([f'"{v}"' if v not in ['null', None] else 'null' for v in allowed])
                descriptions.append(f"- {name}: {allowed_str} - {desc}")
            else:
                descriptions.append(f"- {name}: {feature_type} - {desc}")
        
        return '\n'.join(descriptions)
    
    def generate_json_schema(self) -> str:
        """Generate JSON schema template for response"""
        schema_parts = []
        
        for feature in self.features:
            name = feature['name']
            feature_type = feature['type']
            
            if feature_type == 'int':
                schema_parts.append(f'  "{name}": <integer>')
            elif feature_type == 'boolean':
                schema_parts.append(f'  "{name}": <boolean>')
            elif feature_type.startswith('list'):
                schema_parts.append(f'  "{name}": <array>')
            elif feature_type == 'string':
                schema_parts.append(f'  "{name}": <string>')
            else:
                schema_parts.append(f'  "{name}": <string|null>')
        
        return "{\n" + ",\n".join(schema_parts) + "\n}"
    
    def validate_prediction(self, pred: Dict) -> Tuple[bool, List[str]]:
        """Validate prediction against feature schema"""
        errors = []
        
        for feature in self.features:
            name = feature['name']
            allowed_values = feature.get('allowed_values')
            feature_type = feature['type']
            
            # Check if required field exists
            if name not in pred:
                errors.append(f"Missing required field: {name}")
                continue
            
            value = pred[name]
            
            # Check allowed values for enum types
            if feature_type == 'enum' and allowed_values and value not in allowed_values and value != None:
                errors.append(f"Invalid value for {name}: {value}. Allowed: {allowed_values}")
            
            # Check type
            if value is not None:
                if feature_type == 'int' and not isinstance(value, int):
                    errors.append(f"Invalid type for {name}: expected int, got {type(value)}")
                elif feature_type == 'boolean' and not isinstance(value, bool):
                    errors.append(f"Invalid type for {name}: expected bool, got {type(value)}")
                elif feature_type.startswith('list') and not isinstance(value, list):
                    errors.append(f"Invalid type for {name}: expected list, got {type(value)}")
                elif feature_type == 'string' and not isinstance(value, str):
                    errors.append(f"Invalid type for {name}: expected string, got {type(value)}")
        
        return len(errors) == 0, errors
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return [f['name'] for f in self.features]

def get_analysis_prompt(feature_schema: FeatureSchema) -> str:
    """Generate the unified analysis prompt using features.json."""
    
    feature_descriptions = feature_schema.generate_feature_descriptions()
    json_schema = feature_schema.generate_json_schema()
    
    return f"""You are an expert audio analytics specialist capable of analyzing both voice conversations and music.

You will receive an audio clip that may contain:
1. Phone conversations between customers and agents
2. Music tracks  
3. Mixed audio content

Analyze the audio and provide the following attributes:

FEATURES TO EXTRACT:
{feature_descriptions}

Return ONLY valid JSON in this exact format:
{json_schema}

Rules:
- Use exact values from allowed options
- Return only JSON, no explanations
- If unsure, make best estimate within allowed values
- Use null for features not applicable to the audio type
- All features must be present in response
- For customer_name, extract the actual name mentioned in the call
- For order_number, use "N/A" if no order number is mentioned
- For product_numbers, separate multiple products with " & "
- Listen carefully for specific product codes and order numbers"""

class GeminiProvider:
    """Gemini LLM provider for audio analysis."""
    
    def __init__(self, api_key: str, feature_schema: FeatureSchema):
        self.api_key = api_key
        self.feature_schema = feature_schema
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Audio Analysis App"
        }
    
    def encode_audio(self, audio_path: Path) -> str:
        """Encode audio to base64."""
        with open(audio_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def analyze_audio(self, audio_path: Path, model_name: str, prompt: str) -> Dict:
        """Analyze audio using Gemini model with OPUS format."""
        
        # Use OPUS directly for Gemini
        print(f"  Using OPUS format directly for Gemini")
        pred = self._send_request(audio_path, model_name, prompt)
        
        # Validate prediction against schema
        is_valid, errors = self.feature_schema.validate_prediction(pred)
        if not is_valid:
            print(f"  Validation warnings: {errors}")
        
        return pred
    
    def _send_request(self, audio_path: Path, model_name: str, prompt: str) -> Dict:
        """Send request to OpenRouter API."""
        audio_b64 = self.encode_audio(audio_path)
        
        print(f"  Sending .opus audio to Gemini ({audio_path.name})")
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "audio",
                            "audio": {
                                "data": audio_b64,
                                "format": "audio/ogg"
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(OPENROUTER_URL, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if 'choices' not in result or not result['choices']:
            raise Exception(f"Invalid response: {result}")
        
        content = result['choices'][0]['message']['content']
        
        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise Exception(f"Could not parse JSON: {content}")

def load_ground_truth(labels_path: Path) -> Dict[str, Dict]:
    """Load ground truth labels from JSON file."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    return data.get('labels', {})

def calculate_accuracy(truth: Dict, pred: Dict, feature_schema: FeatureSchema) -> float:
    """Calculate overall accuracy score."""
    if not truth or not pred:
        return 0.0
    
    # Get all feature names from schema
    all_features = feature_schema.get_feature_names()
    
    total_features = 0
    correct_features = 0
    
    for feature_name in all_features:
        if feature_name in truth and feature_name in pred:
            total_features += 1
            if truth[feature_name] == pred[feature_name]:
                correct_features += 1
    
    return correct_features / total_features if total_features > 0 else 0.0

def run_evaluation(model_variant: str = None) -> None:
    """Run evaluation using Gemini."""
    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "raw"
    labels_path = base_dir / "labels.json"
    features_path = base_dir / "features.json"
    
    # Load feature schema
    try:
        feature_schema = FeatureSchema(features_path)
        print(f"Loaded {len(feature_schema.features)} features from {features_path}")
    except FileNotFoundError:
        raise RuntimeError(f"features.json not found at {features_path}")
    
    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    
    # Get model name
    gemini_models = MODEL_CONFIGS["gemini"]
    if model_variant and model_variant in gemini_models:
        model_name = gemini_models[model_variant]
    else:
        # Use first available model
        model_name = list(gemini_models.values())[0]
    
    print(f"Using model: {model_name}")
    print(f"Provider: Gemini")
    print(f"Audio format: OPUS (native)")
    
    # Initialize Gemini provider
    gemini_provider = GeminiProvider(api_key, feature_schema)
    
    # Load labels
    labels = load_ground_truth(labels_path)
    print(f"Loaded {len(labels)} ground truth labels")
    
    # Get prompt (generated from features.json)
    prompt = get_analysis_prompt(feature_schema)
    
    # Find audio files
    audio_files = list(data_dir.glob("*.opus"))
    if not audio_files:
        print("No .opus files found in data/raw/")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process files
    results = []
    accuracy_scores = []
    
    for audio_path in audio_files:
        filename = audio_path.name
        
        if filename not in labels:
            print(f"No ground truth for {filename}, skipping...")
            continue
        
        print(f"Processing {filename}...")
        
        try:
            # Get prediction using Gemini
            pred = gemini_provider.analyze_audio(audio_path, model_name, prompt)
            truth = labels[filename]
            
            # Calculate accuracy for display purposes
            accuracy_score = calculate_accuracy(truth, pred, feature_schema)
            accuracy_scores.append(accuracy_score)
            
            # Store only the three required columns
            result = {
                "filename": filename,
                "truth_json": json.dumps(truth, ensure_ascii=False),
                "pred_json": json.dumps(pred, ensure_ascii=False)
            }
            
            results.append(result)
            
            print(f"  Overall accuracy: {accuracy_score:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    if results:
        output_path = base_dir / "results_gemini.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        # Print summary statistics
        if accuracy_scores:
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            print(f"\nResults saved to {output_path}")
            print(f"CSV contains only: filename, truth_json, pred_json")
            print(f"Average accuracy: {avg_accuracy:.3f}")
            print(f"Processed {len(results)} files successfully")
        else:
            print(f"\nResults saved to {output_path}")
            print(f"CSV contains only: filename, truth_json, pred_json")
            print(f"Processed {len(results)} files successfully")
    else:
        print("No results to save")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gemini Audio Analysis Evaluation")
    parser.add_argument("--variant", choices=["gemini-3-pro-preview", "gemini-2.5-flash"],
                       help="Gemini model variant")
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args.variant)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()