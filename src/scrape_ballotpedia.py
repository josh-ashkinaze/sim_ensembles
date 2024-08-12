import argparse
import datetime
from bs4 import BeautifulSoup
import pandas as pd
import requests
import Levenshtein
import os

"""
Description: Scrapes Ballotpedia state referendums for a given year. This was only tested for 2024 and performance not guaranteed afterwards.

Date: 2024-08-12 19:43:06

Author: Joshua Ashkinaze
"""

FIFTY_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "washington, d.c.", "puerto rico"
]


def scrape_ballotpedia(url):
    """
    The main scraping loop. This was written on (2024-08-12 19:36:55) and performance
    not guaranteed afterwards.

    Args:
        url (str): URL to scrape

    Returns:
        pandas.DataFrame: DataFrame with scraped data
    """
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table', class_='bptable blue')
    dataframes = []
    for table in tables:
        state_header = table.find_previous('h3')
        state_name = state_header.text.strip() if state_header else "State Not Found"
        headers = [header.get_text(strip=True) for header in table.find_all('th')]
        if 'Title' in headers:
            title_index = headers.index('Title')
            headers.insert(title_index + 1, 'URL')
        headers = ['State'] + headers
        rows = []
        for row in table.find_all('tr')[1:]:
            columns = row.find_all('td')
            row_data = [state_name]
            for i, col in enumerate(columns):
                text = col.get_text(strip=True)
                if i == title_index:
                    a_tag = col.find('a')
                    url = a_tag['href'] if a_tag else None
                    row_data.append(text)
                    row_data.append(url)
                else:
                    row_data.append(text)
            rows.append(row_data)
        df = pd.DataFrame(rows, columns=headers)
        dataframes.append(df)
    final_df = pd.concat(dataframes, ignore_index=True)
    final_df.drop_duplicates(inplace=True)
    final_df.columns = [x.lower() for x in final_df.columns]
    return final_df[['state', 'type', 'title', 'url', 'description', 'subject']]


def process_df(df):
    """
    Applies validation functions to the dataframe.

    Args:
        df (pandas.DataFrame): DataFrame to process

    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    df['is_valid_url'] = df['url'].apply(is_valid_url)
    df['is_state'] = df['state'].apply(lambda x: is_a_state(x))
    df['is_valid'] = df['is_valid_url'] & df['is_state']
    return df


def is_valid_url(x):
    """
    Sometimes the URLs are not parsed correctly from scraping. Check if they are valid.

    Args:
        x (str): URL to check

    Returns:
        int: 1 if valid, 0 if not
    """
    return 1 if x.startswith('http://') or x.startswith('https://') else 0


def is_a_state(x, fifty_states=FIFTY_STATES, max_dist=3):
    """
    Check if a string is a state name or at least close to it.

    Args:
        x (str): String to check
        fifty_states (list): List of state names
        max_dist (int): Maximum Levenshtein distance to consider a match

    Returns:
        int: 1 if valid, 0 if not
    """
    x = x.lower()
    for state in fifty_states:
        if Levenshtein.distance(x, state) <= max_dist:
            return 1
    return 0


def main(args):
    """
    Main function to handle workflow.

    Args:
        args (Namespace): Command line arguments
    """
    df = scrape_ballotpedia(args.url)
    df = process_df(df)

    if args.add_dt:
        base_dir = os.path.dirname(args.fn)
        base_filename = os.path.basename(args.fn)
        datetime_prefix = datetime.datetime.now().strftime('%Y-%m-%d_')
        args.fn = os.path.join(base_dir, datetime_prefix + base_filename)

    df.to_csv(args.fn, index=False)
    print(f"Data saved to {args.fn}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape ballot measures from Ballotpedia.')
    parser.add_argument('--url', type=str, default="https://ballotpedia.org/2024_ballot_measures", help='URL to scrape')
    parser.add_argument('--fn', type=str, default='ballotpedia_data.csv', help='Output filename')
    parser.add_argument('--add_dt', action='store_true', help='Add date to filename')
    args = parser.parse_args()
    main(args)
