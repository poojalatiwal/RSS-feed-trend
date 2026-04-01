import requests
from bs4 import BeautifulSoup
import logging
import ray

logger = logging.getLogger(__name__)

# Initialize Ray (safe init)
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)


#  PARALLEL SCRAPER TASK
@ray.remote
def scrape_single_url(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {url: ""}

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=' ')

        # Clean text
        lines = [line.strip() for line in text.splitlines()]
        chunks = []
        for line in lines:
            for phrase in line.split("  "):
                phrase = phrase.strip()
                if phrase:
                    chunks.append(phrase)

        cleaned_text = "\n".join(chunks)

        return {url: cleaned_text[:10000]}

    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return {url: ""}


# -----------------------------
# 🔹 MAIN AGENT
# -----------------------------
class ScraperAgent:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

    def scrape_urls(self, urls):
        """
        Parallel scraping using Ray
        Returns: {url: cleaned_text}
        """
        if not urls:
            return {}

        logger.info(f"Scraping {len(urls)} URLs in parallel...")

        # Create parallel tasks
        tasks = [scrape_single_url.remote(url, self.headers) for url in urls]

        # Collect results
        results_list = ray.get(tasks)

        # Merge results into single dict
        results = {}
        for item in results_list:
            results.update(item)

        return results


# -----------------------------
# 🔹 TEST
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scraper = ScraperAgent()

    test_urls = [
        "https://www.example.com",
        "https://www.bbc.com/news",
        "https://www.cnn.com"
    ]

    data = scraper.scrape_urls(test_urls)

    for url, content in data.items():
        print(f"\n--- {url} ---\n{content[:300]}")