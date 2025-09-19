import requests
from bs4 import BeautifulSoup

def duckduckgo_search(query, max_results=5):
    """
    Performs a DuckDuckGo search and returns a list of (title, href) tuples.
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query}

    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.find_all('a', class_='result__a', limit=max_results):
            title = result.get_text()
            href = result.get('href')
            if href:
                results.append((title, href))
        return results
    except requests.RequestException as e:
        print(f"[Error during search: {e}]")
        return []

def fetch_and_clean_page(url):
    """
    Fetches the content of a URL and returns a cleaned snippet of the main text.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find the main content paragraphs, this is a heuristic for Wikipedia
        # and may need adjustment for other sites.
        paragraphs = soup.find_all('p')
        
        # Concatenate the text of the first few paragraphs to form a snippet.
        snippet = "\n".join([p.get_text() for p in paragraphs[:5]])
        
        # Limit snippet length
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "..."
            
        return snippet if snippet else "[Could not extract a meaningful snippet.]"

    except Exception as e:
        return f"[Error fetching or parsing page: {e}]"

def web_search(query: str) -> str:
    """
    Performs a web search for a query, prioritizing Wikipedia, 
    and returns a clean snippet from the best result.
    """
    print(f"Performing web search for: {query}")
    search_query = f"{query} site:en.wikipedia.org"
    search_results = duckduckgo_search(search_query, max_results=3)

    if not search_results:
        # Fallback to a general search if no Wikipedia results are found
        print("No Wikipedia results, falling back to general search.")
        search_results = duckduckgo_search(query, max_results=1)

    if not search_results:
        return "No search results found."

    # Fetch the content of the first result
    best_title, best_link = search_results[0]
    print(f"Fetching content from: {best_title} ({best_link})")
    
    snippet = fetch_and_clean_page(best_link)
    
    return snippet

if __name__ == '__main__':
    # Example usage:
    test_query = "LangChain"
    result_snippet = web_search(test_query)
    print("\n--- Search Result Snippet ---")
    print(result_snippet)
    print("---------------------------\\n")

    test_query_2 = "AGI"
    result_snippet_2 = web_search(test_query_2)
    print("\n--- Search Result Snippet ---")
    print(result_snippet_2)
    print("---------------------------\\n")
