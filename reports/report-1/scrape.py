"""

Course: Understanding and Building LLMs
Report 1: Developing a simple data pre-processing pipeline

Scraping data from a website and a pdf-file. Both seem useful
techniques for data collection. The files are saved in the
current directory.

Christian Gustavsson, christian.gustavsson@liu.se

"""

import requests
import os

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader


def scrape_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.prettify()

def scrape_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    raw_data = "\n".join(page.page_content for page in pages)

    return raw_data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Scraping data from a website
    url = "https://en.wikipedia.org/wiki/Computer_security"
    raw_data = scrape_url(url)

    output_path = os.path.join(script_dir, "raw_website_data.txt")
    with open(output_path, "w") as f:
        f.write(raw_data)

    # Scraping data from a pdf-file
    pdf_path = "nationell-strategi-for-cybersakerhet-2025-2029.pdf"
    raw_data = scrape_pdf(pdf_path)

    output_path = os.path.join(script_dir, "raw_pdf_data.txt")
    with open(output_path, "w") as f:
        f.write(raw_data)

if __name__ == "__main__":
    main()