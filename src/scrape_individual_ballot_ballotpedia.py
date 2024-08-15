"""
Author: Joshua Ashkinaze

Description: Scrapes individual ballot measures from Ballotpedia. This script can be used to scrape detailed information
about individual ballot measures by providing a URL, a text file with URLs, or a CSV file with a 'url' column.

Date: 2024-08-15 15:30:46
"""



import argparse
import requests
from bs4 import BeautifulSoup, NavigableString
import re
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import os
import datetime


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_html_content(url):
    """
    Fetch HTML content from a given URL with retry logic.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: The HTML content of the page.

    Raises:
        requests.RequestException: If an error occurs while fetching the URL.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def clean_text(text):
    """
    Clean and format the input text.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned and formatted text.
    """
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
    return text.strip()


def extract_yes_means(soup):
    """
    Extract the 'yes means' information from the soup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the page.

    Returns:
        str: The 'yes means' information, or None if not found.
    """
    yes_box = soup.find('table', style=lambda value: value and 'border-left: 10px solid #007F00' in value)
    if yes_box:
        return clean_text(yes_box.get_text())
    return None


def extract_no_means(soup):
    """
    Extract the 'no means' information from the soup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the page.

    Returns:
        str: The 'no means' information, or None if not found.
    """
    no_box = soup.find('table', style=lambda value: value and 'border-left: 10px solid #BF0000' in value)
    if no_box:
        return clean_text(no_box.get_text())
    return None


def process_element_text(element):
    """
    Process the text of an HTML element, handling links and other content.

    Args:
        element (bs4.element.Tag): The BeautifulSoup element to process.

    Returns:
        str: The processed and cleaned text of the element.
    """
    text = []
    for content in element.contents:
        if isinstance(content, NavigableString):
            text.append(str(content))
        elif content.name == 'a':
            text.append(f" {content.get_text()} ")
        else:
            text.append(content.get_text())
    return clean_text(''.join(text))


def extract_overview(soup):
    """
    Extract the overview information from the soup object.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the page.

    Returns:
        str: The overview information, or None if not found.
    """
    overview_header = soup.find('span', {'class': 'mw-headline', 'id': 'Overview'})
    if overview_header:
        overview_content = []
        for sibling in overview_header.find_parent().find_next_siblings():
            if sibling.name == 'h2':
                break
            if sibling.name in ['p', 'ul']:
                overview_content.append(process_element_text(sibling))
        return ' '.join(overview_content)
    return None


def scrape_ballot_measure(url):
    """
    Scrape ballot measure information from a given URL.

    Args:
        url (str): The URL of the ballot measure page.

    Returns:
        dict: A dictionary containing 'yes_means', 'no_means', and 'overview' information.
    """
    html_content = fetch_html_content(url)
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')

    yes_means = extract_yes_means(soup)
    no_means = extract_no_means(soup)
    overview = extract_overview(soup)

    return {
        'yes_means': yes_means,
        'no_means': no_means,
        'overview': overview
    }


def process_urls(urls):
    """
    Process a list of URLs and scrape ballot measure information.

    Args:
        urls (list): A list of URLs to process.

    Returns:
        pd.DataFrame: A DataFrame containing the scraped information.
    """
    results = []
    total_urls = len(urls)

    for i, url in enumerate(urls, 1):
        result = scrape_ballot_measure(url)
        if result:
            result['url'] = url
            result['ballot_no'] = i
            results.append(result)

        if i % max(1, total_urls // 10) == 0:
            print(f"Progress: {i}/{total_urls} URLs processed ({i / total_urls * 100:.1f}%)")

        time.sleep(1)  # Add a small delay between requests

    return pd.DataFrame(results)


def main():
    """
    Main function to run the ballot measure scraper.
    """
    parser = argparse.ArgumentParser(description="Scrape ballot measures from Ballotpedia.")
    parser.add_argument("input", help="URL, text file with URLs, or CSV file with a 'url' column")
    parser.add_argument("fn", help="Output filename")
    parser.add_argument("--add_dt", action="store_true", help="Add date prefix to filename")
    args = parser.parse_args()

    input_data = args.input

    if args.add_dt:
        base_dir = os.path.dirname(args.fn)
        base_filename = os.path.basename(args.fn)
        datetime_prefix = datetime.datetime.now().strftime('%Y-%m-%d_')
        args.fn = os.path.join(base_dir, datetime_prefix + base_filename)

    if input_data.startswith('http'):
        # c1: Single URL
        urls = [input_data]
    elif input_data.endswith('.txt'):
        # c2: Text file with URLs
        with open(input_data, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]
    elif input_data.endswith('.csv'):
        # c3: CSV file with 'url' column
        df = pd.read_csv(input_data)
        if 'url' not in df.columns:
            raise ValueError("CSV file must contain a 'url' column")
        urls = df['url'].tolist()
    else:
        raise ValueError(
            "Invalid input. Provide a URL, a .txt file with URLs, or a .csv file with a 'url' column.")

    results_df = process_urls(urls)
    print(results_df)

    # Save results to CSV
    results_df.to_csv(args.fn, index=False)
    print(f"Results saved to {args.fn}")


if __name__ == "__main__":
    main()