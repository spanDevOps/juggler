"""
NVIDIA NIM Model Scraper using Tavily API
Extracts model information from build.nvidia.com pages using Tavily's extract endpoint
"""

import os
import json
import time
from tavily import TavilyClient


def scrape_nvidia_model_with_tavily(url, api_key):
    """
    Scrape a single NVIDIA model page using Tavily API.
    
    Args:
        url: Full URL to the model page
        api_key: Tavily API key
    
    Returns:
        dict: Extracted model information
    """
    print(f"Scraping with Tavily: {url}")
    
    try:
        # Initialize Tavily client
        client = TavilyClient(api_key=api_key)
        
        print("  Calling Tavily API...")
        
        # Extract content using basic depth first (faster)
        response = client.extract(
            urls=[url],
            include_images=False,
            extract_depth="basic"  # Start with basic, faster
        )
        
        print("  Received response from Tavily")
        
        # Parse the response
        if response and 'results' in response and len(response['results']) > 0:
            result = response['results'][0]
            
            model_info = {
                'url': url,
                'model_id': url.split('/')[-1],
                'publisher': url.split('/')[-2],
                'raw_content': result.get('raw_content', ''),
                'content_length': len(result.get('raw_content', ''))
            }
            
            # Extract structured info from content
            content = result.get('raw_content', '').lower()
            
            # Look for context window
            import re
            context_patterns = [
                r'(\d+[km])\s*(?:token|context)',
                r'context.*?(\d+[km])',
                r'(\d+,?\d*)\s*tokens'
            ]
            for pattern in context_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    model_info['context_window'] = match.group(1)
                    break
            
            # Check for pricing
            if 'free' in content or '$0' in content:
                model_info['pricing'] = 'free'
            elif '$' in content:
                price_match = re.search(r'\$[\d.]+', content)
                if price_match:
                    model_info['pricing'] = price_match.group(0)
            
            # Extract capabilities
            capabilities = []
            capability_keywords = {
                'streaming': ['stream', 'streaming'],
                'vision': ['vision', 'image', 'visual', 'multimodal', 'vlm'],
                'tool_calling': ['tool', 'function calling', 'function call'],
                'reasoning': ['reasoning', 'chain-of-thought', 'cot', 'think'],
                'code': ['code', 'coding', 'programming', 'coder'],
                'multilingual': ['multilingual', 'multi-language', 'languages'],
                'json': ['json', 'structured output'],
                'embedding': ['embed', 'embedding', 'vector'],
                'translation': ['translate', 'translation'],
                'detection': ['detect', 'detection', 'object detection'],
            }
            
            for cap, keywords in capability_keywords.items():
                if any(kw in content for kw in keywords):
                    capabilities.append(cap)
            
            model_info['capabilities'] = capabilities
            
            print(f"✅ Successfully scraped: {model_info['model_id']}")
            return model_info
        else:
            print(f"❌ No results from Tavily for {url}")
            return {'url': url, 'error': 'No results from Tavily'}
            
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return {'url': url, 'error': str(e)}


def test_single_model():
    """Test Tavily scraper with an LLM model"""
    # Get API key from environment
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        print("❌ Error: TAVILY_API_KEY environment variable not set")
        print("Please set it with: set TAVILY_API_KEY=tvly-your-key")
        return None
    
    # Test with an actual LLM model (DeepSeek V3.1)
    test_url = "https://build.nvidia.com/deepseek-ai/deepseek-v3_1"
    
    print("=" * 60)
    print("Testing NVIDIA Tavily Scraper")
    print("=" * 60)
    
    result = scrape_nvidia_model_with_tavily(test_url, api_key)
    
    print("\n" + "=" * 60)
    print("EXTRACTED DATA:")
    print("=" * 60)
    print(json.dumps({k: v for k, v in result.items() if k != 'raw_content'}, indent=2))
    
    # Save full result with raw content
    with open("nvidia_streampetr_tavily.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print("\n💾 Saved full data to nvidia_streampetr_tavily.json")
    
    # Save just the raw content for inspection
    if 'raw_content' in result:
        with open("nvidia_streampetr_content.txt", 'w', encoding='utf-8') as f:
            f.write(result['raw_content'])
        print("💾 Saved raw content to nvidia_streampetr_content.txt")
    
    return result


if __name__ == "__main__":
    test_single_model()
