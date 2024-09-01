"""
Author: Joshua Ashkinaze

Description: Fetch solar panel responses.

The basic idea is this to simulate a focus group of conservatives who say what they want in a solar panel product. And
then the mod takes those responses and comes up with something based on that.
"""

import re
import unicodedata
import pandas as pd
from tqdm import tqdm
from plurals.agent import Agent
from plurals.deliberation import Moderator, Ensemble
from pprint import pprint
import ftfy
import html


def pretty_print_df(df):
    for idx, row in df.iterrows():
        print(f"Index: {row['idx']}")
        print("=" * 20)
        print("Focus Group Response:")
        pprint(row['clean_focus_group_final'])
        print("=" * 20)
        print("ZeroShot Response:")
        pprint(row['clean_zero'])
        print("=" * 20)
        print()


def clean_text(text):

    text = strip_response_string(text)

    text = unicodedata.normalize('NFKC', text)

    replacements = {
        '—': '&mdash;',  # em dash
        '–': '&ndash;',  # en dash
        '"': '&ldquo;',  # left double quotation mark
        '"': '&rdquo;',  # right double quotation mark
        ''': '&lsquo;',  # left single quotation mark
        ''': '&rsquo;',  # right single quotation mark
        '…': '&hellip;',  # ellipsis
    }

    for char, entity in replacements.items():
        text = text.replace(char, entity)

    return text


def strip_response_string(response):
    """
    Strips the "Response" string from the beginning of a response if it exists.
    """
    match = re.match(r'Response \d+: (.*)', response)
    if match:
        r = match.group(1).strip()
    else:
        r = response.strip()

    r = r.replace('"', '').replace("'", '').replace("''", "")
    return r.strip()


def main():
    N = 15
    N_WORDS = 50
    MODEL = 'gpt-4o'

    data_pts = []
    structure_data = []
    for i in tqdm(range(N)):

        # Focus group task and participants
        ############################
        focus_group_task = f"What specific product details for a solar panel company would resonate with you personally? Be very specific; you are in a focus group. Answer in 20 words."
        focus_group_participants = [Agent(model=MODEL, task=focus_group_task, ideology='conservative') for _ in
                                    range(10)]

        moderator = Moderator(model=MODEL,
                              system_instructions="You are an expert copywriter for an ad agency.",
                              task="You are overseeing a focus group discussing what products would resonate with them for the solar panel category.",
                              combination_instructions=f"Here are focus group responses: \n<start>${{previous_responses}}<end>. Now based on the specifics of these responses, come up with a specific product for a solar panel company that would resonate with the focus group members. Be very specific. Answer in {N_WORDS} words only.")

        ensemble = Ensemble(agents=focus_group_participants, moderator=moderator)
        ensemble.process()
        print(ensemble.moderator.history)

        ensemble_response = ensemble.final_response
        ############################

        # Default
        ############################
        zero_shot_task = f"Come up with a specific product for a solar panel company that would resonate with conservatives. Be very specific. Answer in {N_WORDS} words only."
        zero_shot = Agent(model=MODEL,
                        system_instructions="You are an expert copywriter for an ad agency.",
                        task=zero_shot_task)
        zero_shot_response = zero_shot.process()

        data_pt = {'focus_group_final': ensemble_response, 'zero_shot': zero_shot_response, 'idx': i + 1,
                   'focus_group_all': ensemble.responses}
        structure_info = ensemble.info
        structure_info.update({'idx':i+1})
        data_pts.append(data_pt)
        structure_data.append(structure_info)
        ############################

    df = pd.DataFrame(data_pts)
    df['clean_focus_group_final'] = df['focus_group_final'].apply(clean_text)
    df['clean_zero'] = df['zero_shot'].apply(clean_text)
    df.drop(columns=['focus_group_all']).to_csv(f"../data/raw/solar_panels_conservatives_gpt-4o_script_{N_WORDS}.csv", index=False)

    structure_df = pd.DataFrame(structure_data)
    structure_df.to_json(f"../data/raw/solar_panels_conservatives_gpt-4o_script_{N_WORDS}_structure.jsonl", orient='records', lines=True)

    print("Data processing complete.")

    pretty_print_df(df)


if __name__ == "__main__":
    main()