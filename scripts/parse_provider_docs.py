"""
Parse provider documentation using Jugglerr's own LLMs.

This script takes raw HTML/text from provider docs and uses Jugglerr
to extract model information intelligently.
"""

import json
import sys
from jugglerr import LLMJugglerr


def parse_models_from_docs(content: str, provider: str) -> dict:
    """
    Use Jugglerr to parse provider documentation and extract model info.
    
    Args:
        content: Raw HTML or text content from provider docs
        provider: Provider name (groq, cerebras, mistral, etc.)
    
    Returns:
        dict: Parsed model information
    """
    jugglerr = LLMJugglerr()
    
    prompt = f"""
You are analyzing documentation for {provider} LLM provider.

Extract ALL models mentioned in this documentation. For each model, identify:
1. Model name/ID (exact string used in API)
2. Capabilities (streaming, vision, tool_calling, json_object, reasoning, multilingual, etc.)
3. Context window size (in tokens)
4. Power level (super for large/flagship models, regular for smaller/efficient models)
5. Any special notes (deprecated, beta, etc.)

Documentation content:
{content[:15000]}  # Limit to avoid token limits

Return ONLY a JSON array with this structure:
[
  {{
    "model_name": "exact-model-id",
    "power": "super" or "regular",
    "capabilities": ["streaming", "vision", "tool_calling"],
    "context_window": 128000,
    "notes": "optional notes"
  }}
]

Be thorough - extract ALL models mentioned. If context window is not specified, use null.
Return ONLY the JSON array, no other text.
"""
    
    try:
        response = jugglerr.juggle(
            messages=[{"role": "user", "content": prompt}],
            preferred_provider='groq',  # Fast and free
            temperature=0.1,  # Low temp for accuracy
            max_tokens=4000
        )
        
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        models = json.loads(response)
        return {
            'provider': provider,
            'models': models,
            'success': True
        }
    
    except Exception as e:
        return {
            'provider': provider,
            'error': str(e),
            'success': False
        }


def compare_with_existing(provider: str, new_models: list) -> dict:
    """
    Compare newly found models with existing models.py database.
    
    Returns:
        dict: {
            'new_models': [...],
            'changed_models': [...],
            'removed_models': [...]
        }
    """
    from jugglerr.models import MODEL_DATABASE
    
    existing = MODEL_DATABASE.get(provider, {})
    existing_names = set(existing.keys())
    new_names = set(m['model_name'] for m in new_models)
    
    return {
        'new_models': [m for m in new_models if m['model_name'] not in existing_names],
        'changed_models': [],  # TODO: detect capability changes
        'removed_models': list(existing_names - new_names),
        'total_existing': len(existing_names),
        'total_found': len(new_names)
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python parse_provider_docs.py <provider> <content_file>")
        sys.exit(1)
    
    provider = sys.argv[1]
    content_file = sys.argv[2]
    
    with open(content_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Parsing {provider} documentation...")
    result = parse_models_from_docs(content, provider)
    
    if result['success']:
        print(f"\nFound {len(result['models'])} models:")
        for model in result['models']:
            print(f"  - {model['model_name']}")
        
        # Compare with existing
        comparison = compare_with_existing(provider, result['models'])
        
        print(f"\nComparison with existing database:")
        print(f"  Existing models: {comparison['total_existing']}")
        print(f"  Found in docs: {comparison['total_found']}")
        
        if comparison['new_models']:
            print(f"\n  NEW MODELS ({len(comparison['new_models'])}):")
            for model in comparison['new_models']:
                print(f"    + {model['model_name']}")
        
        if comparison['removed_models']:
            print(f"\n  REMOVED/DEPRECATED ({len(comparison['removed_models'])}):")
            for model_name in comparison['removed_models']:
                print(f"    - {model_name}")
        
        if not comparison['new_models'] and not comparison['removed_models']:
            print("  ✓ Database is up to date!")
        
        # Save full result
        output_file = f"{provider}_models_parsed.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results saved to: {output_file}")
    
    else:
        print(f"Error: {result['error']}")
        sys.exit(1)
