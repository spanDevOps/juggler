"""
Quick script to scrape Cohere URLs provided as arguments.

Usage:
    python scripts/scrape_cohere_urls.py "url1" "url2" "url3"
"""

import os
import sys
import json

def scrape_urls(urls):
    """Scrape URLs using Tavily."""
    try:
        from tavily import TavilyClient
    except ImportError:
        print("Installing tavily-python...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tavily-python"])
        from tavily import TavilyClient
    
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        print("❌ Error: TAVILY_API_KEY environment variable not set")
        sys.exit(1)
    
    client = TavilyClient(api_key=api_key)
    
    print(f"Scraping {len(urls)} URLs...")
    print("=" * 70)
    
    all_content = {}
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        try:
            response = client.extract(urls=[url])
            
            if response and 'results' in response and len(response['results']) > 0:
                content = response['results'][0].get('raw_content', '')
                all_content[url] = content
                print(f"✓ Scraped {len(content):,} characters")
            else:
                print("⚠️  No content returned")
                all_content[url] = ""
        except Exception as e:
            print(f"❌ Error: {e}")
            all_content[url] = ""
    
    # Save to file
    output_file = "cohere_scraped_content.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"✅ Saved to: {output_file}")
    print(f"Total content: {sum(len(c) for c in all_content.values()):,} characters")
    
    # Also save as text for easy reading
    text_file = "cohere_scraped_content.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        for url, content in all_content.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"URL: {url}\n")
            f.write(f"{'='*70}\n\n")
            f.write(content)
            f.write(f"\n\n")
    
    print(f"✅ Saved readable version to: {text_file}")
    
    return all_content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/scrape_cohere_urls.py <url1> <url2> ...")
        print("\nExample:")
        print('  python scripts/scrape_cohere_urls.py "https://docs.cohere.com/docs/models"')
        sys.exit(1)
    
    urls = sys.argv[1:]
    scrape_urls(urls)
