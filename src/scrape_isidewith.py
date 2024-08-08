"""
Author: Joshua Ashkinaze

Date: 2024-08-08

Description: This script scrapes political polls from the website 'isidewith.com' based on user-specified sorting options.
It allows the user to specify sorting by 'all', 'popular', 'new', or 'local' categories.

Usage:
  python scrape_polls.py --sort_option [OPTION] --fn [FILENAME] [--add_dt]

Arguments:
  --sort_option : Specify the sorting option for the polls (default: 'popular').
                  Options include 'all', 'popular', 'new', or 'local'.
  --fn          : Specify the filename for the output CSV (default: 'poll_data.csv').
  --add_dt      : Add a datetime prefix to the filename (optional).
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import logging
import argparse
from datetime import datetime

def scrape_polls(sort_option="popular"):
    """
    Scrapes isidewith political issues, optionally sorted by 'all', 'popular', 'new', or 'local'.

    Args:
      sort_option (str): A string in the list ['all', 'popular', 'new', 'local']. Default is 'popular'

    Returns:
      A pandas DataFrame with columns ['initial_order', 'issue', 'question', 'yes_percent', 'no_percent', 'votes'].

    Notes:
      - Initial order is ascending so initial_order==0 and sort_option=='popular' means the most popular.
      - It appears isidewith does not update the 'local' poll listing anymore.
    """
    assert sort_option in ['all', 'popular', 'new', 'local'], "Sort option has to be in ['all', 'popular', 'new', 'local']"
    url = f'https://www.isidewith.com/polls/{sort_option}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    data = []
    polls = soup.find_all('div', class_='poll')
    for poll in polls:
        try:
            issue = poll.find('div', class_='img').find('p').find('span').text
            question = poll.find('p', class_='question').text
            yes_votes = poll.find('div', class_='yes').text.split()[0]
            no_votes = poll.find('div', class_='no').text.split()[0]
            total_votes = poll.find('div', class_='count').text.split()[0].replace(',', '')

            data.append([issue, question, yes_votes, no_votes, total_votes])
        except Exception as e:
            logging.error(f"Failed to process a poll: {e}")
            continue
    df = pd.DataFrame(data, columns=['issue', 'question', 'yes_percent', 'no_percent', 'votes'])
    df['initial_order'] = [i for i in range(len(df))]
    df['yes_percent'] = df['yes_percent'].astype(int)
    df['no_percent'] = df['no_percent'].astype(int)
    df['votes'] = df['votes'].astype(int)

    return df

def main():
    parser = argparse.ArgumentParser(description='Scrape political polls from isidewith.')
    parser.add_argument('--sort_option', type=str, default='popular',
                        help='Sort option for the polls: all, popular, new, or local')
    parser.add_argument('--fn', type=str, default='poll_data.csv',
                        help='Filename for the output CSV')
    parser.add_argument('--add_dt', action='store_true',
                        help='Add datetime prefix to the filename')

    args = parser.parse_args()

    if args.add_dt:
        base_dir = os.path.dirname(args.fn)
        base_filename = os.path.basename(args.fn)
        datetime_prefix = datetime.now().strftime('%Y-%m-%d_')
        args.fn = os.path.join(base_dir, datetime_prefix + base_filename)

    df = scrape_polls(args.sort_option)
    df.to_csv(args.fn, index=False)
    print(f"Data scraped and saved to {args.fn}")

if __name__ == "__main__":
    main()