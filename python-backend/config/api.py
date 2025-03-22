import requests

def fetch_urls():
    """Fetch URLs from backend API and return them"""
    try:
        response = requests.get('http://localhost:5000/api/urls')
        
        if response.status_code == 200:
            data = response.json()
            urls = data.get('urls', [])
            return urls
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
            return []
            
    except Exception as e:
        print(f"Error fetching URLs: {e}")
        return []

# Export the first URL for easy access by other modules
def get_csv_url():
    """Get the first URL from the API (CSV file URL)"""
    urls = fetch_urls()
    if urls and len(urls) > 0:
        return urls[0]
    return None

if __name__ == "__main__":
    # Example of accessing the URL as a variable
    csv_url = get_csv_url()
    if csv_url:
        print(f"{csv_url}")