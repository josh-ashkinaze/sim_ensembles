"""
Author: Joshua Ashkinaze

Description: Simple script to merge the baseline ballot issues with the detailed ballot issues, and calculcate some word counts.
Then it creates a prompt for each ballot issue.

Date: 2024-08-27 16:42:35
"""



import pandas as pd
import textstat
import re
import logging
import os
from pprint import pprint

logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO, format='%(asctime)s: %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S', force=True)



def pretty_print_prompts(df, prompt_col):
    for _, row in df.iterrows():
        print("="*20)

        pprint(row[prompt_col])
        print()
        print("="*20)

def make_prompt(row):
    """
    Returns a formatted prompt for a ballot issue.

    Args:
    - row (pd.Series): A row from the dataframe

    Returns:
    - str: The formatted prompt
    """
    return f"DESCRIPTION\n{row['description']}\n\nVOTING\n-{row['yes_means']}\n\n-{row['no_means']}\n\nDETAILED OVERVIEW\n{row['overview']}"

def flesch_kincaid(text):
    """
    Returns the Flesch-Kincaid grade level of a text.

    Args:
    - text (str): The text to analyze

    Returns:
    - float: The Flesch-Kincaid grade level
    """
    return textstat.flesch_kincaid_grade(text) if text else None

def clean_text(text):
    """
    Cleans up text
    - Removes bracketed citations
    - Fixes U. S. to U.S.

    Args:
    - text (str): The text to clean

    Returns:
    - str: The cleaned text
    """
    text = text.replace("U. S.", "U.S.")
    text = re.sub(r'\[\d+\]', '', text) # try to remove the bracketed citations
    return text

def sample_text(df, flesch_q, wc_q):
    """Helper function for sampling a text from the dataframe by quatiles of flesch and wc"""
    return df.query(f'description_wc_q == {wc_q} & description_fk_q == {flesch_q}').sample(1)['description'].values[0]

def process_dataframe(df, text_column):
    """
    Add some word count and readability metrics to the dataframe.

    Args:
    - df (pd.DataFrame): The dataframe to process
    - text_column (str): The column containing the text to process

    Returns:
    - pd.DataFrame: The processed dataframe
    """
    df[f'{text_column}_wc'] = df[text_column].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
    df[f'{text_column}_wc_q'] = pd.qcut(df[f'{text_column}_wc'], 4, labels=False) + 1
    df[f'{text_column}_fk'] = df[text_column].apply(flesch_kincaid)
    df[f'{text_column}_fk_q'] = pd.qcut(df[f'{text_column}_fk'], 4, labels=False) + 1
    return df

def main():
    ballots = pd.read_csv('../data/raw/2024-08-15_ballot_issues.csv')
    logging.info("Initial ballots: %d", len(ballots))

    # filter for valid metadata
    ballots = ballots.query('is_valid == 1')
    logging.info("Ballots with valid metadata: %d", len(ballots))

    # clean detailed ballots
    detailed_ballots = pd.read_csv("../data/raw/2024-08-15_detailed_ballot_issues.csv")
    detailed_ballots = detailed_ballots.dropna(subset=['yes_means', 'no_means', 'overview'])

    merged = pd.merge(ballots, detailed_ballots, on='url', how='inner', suffixes=('', '_detailed'))
    logging.info("Ballots with detailed information: %d", len(merged))

    merged['overview'] = merged['overview'].apply(clean_text)
    merged = process_dataframe(merged, 'overview')
    merged['prompt'] = merged.apply(make_prompt, axis=1)
    merged.to_csv('../data/processed/ballot_prompts.csv',)

    sample = merged.sample(30, random_state=42)
    sample.to_csv('../data/processed/sample_ballot_prompts.csv')
    pretty_print_prompts(merged, 'prompt')

    return merged

if __name__ == "__main__":
    final_df = main()
