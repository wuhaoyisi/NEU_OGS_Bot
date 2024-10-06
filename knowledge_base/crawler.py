import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
from playwright.sync_api import sync_playwright

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to normalize URLs (strip query parameters and trailing slashes)
def normalize_url(url: str) -> str:
    parsed_url = urlparse(url)
    normalized_url = parsed_url._replace(query="", fragment="").geturl()
    if normalized_url.endswith('/'):
        normalized_url = normalized_url[:-1]
    return normalized_url

# Helper function to check if a URL is valid and starts with the base URL
def is_valid_url(url: str, base_url: str) -> bool:
    return url.startswith(base_url)

# Function to extract and render content with Playwright for JS-rendered pages
def get_rendered_content(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state('networkidle')
        content = page.content()
        browser.close()
    return content

# Function to find the next available folder to store results
def find_next_folder(base_folder: str = "crawler_txt_") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    
    i = 0
    while True:
        folder_name = f"{base_folder}{i:02d}"
        folder_path = os.path.join(script_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        i += 1

# Function to process the page content and save it to a text file
def process_page(url: str, soup: BeautifulSoup, output_folder: str) -> None:
    main_content = soup.find('main') or soup.body  # Fall back to body if main is not found
    text_content = main_content.get_text(separator='\n', strip=True) if main_content else "No content found"
    
    text_content = f"URL: {url}\n\n{text_content}\n\n{'='*80}\n\n"
    
    file_path = os.path.join(output_folder, "combined_content.txt")
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text_content)

# Function to crawl through the site
def crawl(url: str, base_url: str, output_folder: str) -> None:
    visited = set()
    to_visit = [url]
    total_crawled = 0

    while to_visit:
        current_url = to_visit.pop(0)
        normalized_url = normalize_url(current_url)

        if normalized_url in visited:
            continue

        visited.add(normalized_url)

        try:
            content = get_rendered_content(current_url)
            soup = BeautifulSoup(content, 'html.parser')

            process_page(current_url, soup, output_folder)

            total_crawled += 1
            logger.info(f"Successfully crawled: {current_url} (Total: {total_crawled})")

            for link in soup.find_all('a', href=True):
                new_url = urljoin(current_url, link['href'])
                normalized_new_url = normalize_url(new_url)

                if is_valid_url(normalized_new_url, base_url) and normalized_new_url not in visited:
                    to_visit.append(normalized_new_url)

        except Exception as e:
            logger.error(f"Error crawling {current_url}: {e}")

# Main execution point
if __name__ == "__main__":
    base_url = "https://international.northeastern.edu/ogs/"
    base_domain = urlparse(base_url).netloc

    output_folder = find_next_folder()

    urls_to_crawl = [base_url]

    for url in urls_to_crawl:
        crawl(url, base_url, output_folder)

    logger.info(f"All content has been combined into a single file: {os.path.join(output_folder, 'combined_content.txt')}")
