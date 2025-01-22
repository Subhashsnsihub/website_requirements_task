# Web Scraper and Q&A System

This repository contains a Python-based Web Scraper and Q&A system designed to crawl websites, extract important information such as contact details, metrics, and images, and provide answers to questions based on the extracted data. It uses the OpenAI API for question answering, BeautifulSoup for web scraping, and Tesseract for optical character recognition (OCR) to extract text from images.

## Features

- **Web Scraping**: Crawl websites and extract relevant data like page title, metadata, content, headings, links, and tables.
- **OCR for Image Text**: Extract text from images using Tesseract OCR.
- **Contact Information Extraction**: Detect emails, phone numbers, social media handles, addresses, and working hours from scraped text.
- **Metrics Extraction**: Extract numerical metrics such as number of years, students, faculty, labs, and other metrics from the page content.
- **Question Answering**: Use OpenAI's API to answer user questions based on the extracted website data.
- **Crawling Control**: Limit the number of pages crawled and ensure that links belong to the same domain.

## Prerequisites

- Python 3.7+
- OpenAI API key (for question answering)
- Tesseract OCR (if you want to enable OCR functionality for image text extraction)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/web-scraper-qa.git
    cd web-scraper-qa
    ```

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Install Tesseract OCR (for Windows, Mac, and Linux):
    - [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract/wiki)

4. Set up your OpenAI API key:
    - Add your API key in the script or pass it to the WebScraperQA class.

## Usage

### Initialize the Web Scraper and Q&A System

```python
from openai import OpenAI
from web_scraper_qa import WebScraperQA

# Replace with your OpenAI API key
api_key = 'your-openai-api-key'

# Initialize the WebScraperQA system
scraper_qa = WebScraperQA(api_key=api_key)
