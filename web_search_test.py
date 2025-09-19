import requests
from bs4 import BeautifulSoup

def duckduckgo_search(query, max_results=5):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query}

    response = requests.post(url, headers=headers, data=data)
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for result in soup.find_all('a', class_='result__a', limit=max_results):
        title = result.get_text()
        href = result['href']
        results.append((title, href))

    return results

def fetch_page_snippet(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=5)
        return response.text
    except Exception as e:
        return f"[Error fetching page: {e}]"

# Run search
search_results = duckduckgo_search("who is liveoverflow wikipedia")

# Fetch and print content
for title, link in search_results:
    print(f"Title: {title}")
    print(f"URL: {link}")
    snippet = fetch_page_snippet(link)
    print(f"Snippet: {snippet}\n")
    break
