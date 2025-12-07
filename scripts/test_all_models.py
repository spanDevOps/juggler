#!/usr/bin/env python3
"""
Comprehensive test of ALL Juggler models.
Tests each model with both regular and streaming requests.
"""
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from juggler import Juggler
from juggler.models import MODEL_DATABASE

# Test prompts
TEST_PROMPTS = {
    'simple': "Say 'Hello, I am working!' in one sentence.",
    'reasoning': "What is 2+2? Explain briefly.",
    'creative': "Write a haiku about coding.",
}

def get_all_models():
    """Extract all model IDs from the database."""
    models = []
    for provider, provider_models in MODEL_DATABASE.items():
        for model_id in provider_models.keys():
            models.append({
                'id': model_id,
                'provider': provider,
                'info': provider_models[model_id]
            })
    return models

def test_model_regular(juggler, model_id, provider, prompt):
    """Test a model with regular request - using Juggler as a user would."""
    try:
        start_time = time.time()
        
        # Use Juggler exactly as a user would - just specify the model
        response = juggler.chat(
            messages=[{"role": "user", "content": prompt}],
            preferred_model=model_id,
            max_tokens=100,
            temperature=0.7
        )
        
        elapsed = time.time() - start_time
        
        return {
            'success': True,
            'response': response,
            'elapsed': elapsed,
            'length': len(response)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def test_model_streaming(juggler, model_id, provider, prompt):
    """Test a model with streaming request - using Juggler as a user would."""
    try:
        start_time = time.time()
        chunks = []
        
        # Use Juggler exactly as a user would - just specify the model
        for chunk in juggler.chat_stream(
            messages=[{"role": "user", "content": prompt}],
            preferred_model=model_id,
            max_tokens=100,
            temperature=0.7
        ):
            chunks.append(chunk)
        
        elapsed = time.time() - start_time
        full_response = ''.join(chunks)
        
        return {
            'success': True,
            'response': full_response,
            'elapsed': elapsed,
            'chunks': len(chunks),
            'length': len(full_response)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def main():
    print("=" * 80)
    print("COMPREHENSIVE JUGGLER MODEL TEST")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize Juggler (auto-loads from .env)
    print("Initializing Juggler...")
    juggler = Juggler()
    print("✓ Juggler initialized")
    print()
    
    # Get all models
    all_models = get_all_models()
    print(f"Found {len(all_models)} models across {len(MODEL_DATABASE)} providers")
    print()
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(all_models),
        'models': {}
    }
    
    # Test each model
    for i, model in enumerate(all_models, 1):
        model_id = model['id']
        provider = model['provider']
        
        print(f"\n[{i}/{len(all_models)}] Testing: {model_id}")
        print(f"Provider: {provider}")
        print(f"Power: {model['info'].get('power', 'N/A')}")
        print(f"Capabilities: {', '.join(model['info'].get('capabilities', []))}")
        print("-" * 80)
        
        model_results = {
            'provider': provider,
            'power': model['info'].get('power'),
            'capabilities': model['info'].get('capabilities', []),
            'context_window': model['info'].get('context_window'),
            'tests': {}
        }
        
        # Test 1: Regular request
        print("  [1/2] Testing regular request...")
        regular_result = test_model_regular(juggler, model_id, provider, TEST_PROMPTS['simple'])
        model_results['tests']['regular'] = regular_result
        
        if regular_result['success']:
            print(f"  ✓ SUCCESS ({regular_result['elapsed']:.2f}s, {regular_result['length']} chars)")
            print(f"    Response: {regular_result['response'][:100]}...")
        else:
            print(f"  ✗ FAILED: {regular_result['error_type']}")
            print(f"    Error: {regular_result['error'][:100]}...")
        
        # Test 2: Streaming request (only if model supports streaming)
        if 'streaming' in model['info'].get('capabilities', []):
            print("  [2/2] Testing streaming request...")
            stream_result = test_model_streaming(juggler, model_id, provider, TEST_PROMPTS['simple'])
            model_results['tests']['streaming'] = stream_result
            
            if stream_result['success']:
                print(f"  ✓ SUCCESS ({stream_result['elapsed']:.2f}s, {stream_result['chunks']} chunks)")
                print(f"    Response: {stream_result['response'][:100]}...")
            else:
                print(f"  ✗ FAILED: {stream_result['error_type']}")
                print(f"    Error: {stream_result['error'][:100]}...")
        else:
            print("  [2/2] Streaming not supported - SKIPPED")
            model_results['tests']['streaming'] = {'success': None, 'reason': 'not_supported'}
        
        results['models'][model_id] = model_results
        
        # Small delay between models
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    skipped_tests = 0
    
    for model_id, model_result in results['models'].items():
        for test_type, test_result in model_result['tests'].items():
            total_tests += 1
            if test_result['success'] is True:
                successful_tests += 1
            elif test_result['success'] is False:
                failed_tests += 1
            else:
                skipped_tests += 1
    
    print(f"Total Models: {len(all_models)}")
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
    print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    print(f"Skipped: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
    
    # Save results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Provider breakdown
    print("\n" + "=" * 80)
    print("PROVIDER BREAKDOWN")
    print("=" * 80)
    
    provider_stats = {}
    for model_id, model_result in results['models'].items():
        provider = model_result['provider']
        if provider not in provider_stats:
            provider_stats[provider] = {'total': 0, 'success': 0, 'failed': 0}
        
        provider_stats[provider]['total'] += 1
        
        regular_test = model_result['tests'].get('regular', {})
        if regular_test.get('success'):
            provider_stats[provider]['success'] += 1
        elif regular_test.get('success') is False:
            provider_stats[provider]['failed'] += 1
    
    for provider, stats in sorted(provider_stats.items()):
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{provider:15} | Models: {stats['total']:3} | Success: {stats['success']:3} | Failed: {stats['failed']:3} | Rate: {success_rate:5.1f}%")
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
