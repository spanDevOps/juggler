"""
Quick script to scrape NVIDIA models page using Tavily.

Usage:
    python scripts/scrape_nvidia_models.py
"""

import os
import sys
import json

def scrape_nvidia_models():
    """Scrape NVIDIA models page using Tavily."""
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    try:
        from tavily import TavilyClient
    except ImportError:
        print("Installing tavily-python...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tavily-python"])
        from tavily import TavilyClient
    
    # Get API keys (supports comma-separated multiple keys)
    api_keys_str = os.getenv('TAVILY_API_KEYS')
    if not api_keys_str:
        print("❌ Error: TAVILY_API_KEYS environment variable not set")
        print("Please add TAVILY_API_KEYS to your .env file")
        sys.exit(1)
    
    # Parse all keys
    api_keys = [k.strip() for k in api_keys_str.split(',')]
    print(f"Found {len(api_keys)} API key(s)")
    
    url = "https://build.nvidia.com/models"
    
    print(f"Scraping NVIDIA models page...")
    print("=" * 70)
    print(f"URL: {url}")
    print()
    
    # Try each API key until one works
    last_error = None
    for i, api_key in enumerate(api_keys, 1):
        try:
            print(f"Trying API key {i}/{len(api_keys)}...")
            client = TavilyClient(api_key=api_key)
            response = client.extract(urls=[url])
            
            if response and 'results' in response and len(response['results']) > 0:
                content = response['results'][0].get('raw_content', '')
                print(f"✓ Scraped {len(content):,} characters")
                
                # Save to file
                output_file = "data/nvidia_models_scraped.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'url': url,
                        'content': content,
                        'raw_response': response['results'][0]
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Saved to: {output_file}")
                
                # Also save as text for easy reading
                text_file = "data/nvidia_models_scraped.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"URL: {url}\n")
                    f.write(f"{'='*70}\n\n")
                    f.write(content)
                
                print(f"✅ Saved readable version to: {text_file}")
                print(f"\nTotal content: {len(content):,} characters")
                
                return content
            else:
                print(f"⚠️  No content returned with key {i}")
                last_error = "No content returned"
                
        except Exception as e:
            print(f"❌ Error with key {i}: {e}")
            last_error = str(e)
            if i < len(api_keys):
                print(f"Trying next key...")
            continue
    
    # All keys failed
    print(f"\n❌ All {len(api_keys)} API key(s) failed")
    print(f"Last error: {last_error}")
    return None


if __name__ == "__main__":
    scrape_nvidia_models()
