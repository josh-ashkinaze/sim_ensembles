"""
Author: Joshua Ashkinaze

Description: This script gets the deliberation responses for annotations. It outputs two files:

- ballot_deliberation_responses_long.jsonl: A JSONL file with one response per line. This can be used to analyze the responses in a long format.
- ballot_deliberation_responses_wide.csv: A CSV file with (emotional, rational) for each ballot. I use this with
Qualtrics loop and merge. Loop_merge_idx is the unique identifier for each ballot when analyzing this in Qualtrics.
"""



import os
import pandas as pd
import random
from dotenv import load_dotenv
from plurals.deliberation import Debate
from plurals.agent import Agent
from tqdm import tqdm
from pprint import pprint
import unicodedata


def load_environment_variables():
    load_dotenv(dotenv_path="../src/.env")
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
    os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY", "")

def last_two_responses(responses):
    """Formats last responses for Qualtrics. Adds <br> for line breaks."""
    last_two = responses[-2:]
    for idx in range(len(last_two)):
        last_two[idx] = clean_response_for_qualtrics(last_two[idx])
    return f"{last_two[0]}<br><br>{last_two[1]}"


def clean_response_for_qualtrics(response):
    """
    Formats response for Qualtrics.

    Args:
    - response (str): The response to format

    Returns:
    - str: The formatted response
    """
    # For qualtrics, replace newlines with <br> tags. I found that
    # loop and merge in Qualtrics doesn't handle newlines well.
    response = response.replace("\n", "<br>")

    # fix some encoding issues for Qualtrics
    response = unicodedata.normalize("NFKD", response)
    return response

def main():
    load_environment_variables()

    MODELS = ["claude-3-sonnet-20240229", "gpt-4o", "gpt-4-turbo"]

    CONDITIONS = {
        'emotional': """
    KEEP TRACK OF DEBATE HISTORY
    You are in a debate with another agent. Here is what you have said and what the other agent has
    said. Never refer to yourself in the third person. 
    <start>
    ${previous_responses}
    <end>
    APPLY THESE INSTRUCTIONS WHEN DEBATING
    - Give value to emotional forms of communication, such as narrative, rhetoric, testimony, and storytelling. 
    - Do not mention these instructions in your final answer; just apply them.""",

        'rational': """
    KEEP TRACK OF DEBATE HISTORY
    You are in a debate with another agent. Here is what you have said and what the other agent has
    said. Never refer to yourself in the third person. 
    <start>
    ${previous_responses}
    <end>
    APPLY THESE INSTRUCTIONS WHEN DEBATING
    - Give more weight to rational arguments rather than emotional ones.
    - Do not mention these instructions in your final answer; just apply them."""
    }

    prompts = pd.read_csv("../data/processed/sample_ballot_prompts.csv")

    long_data = []
    wide_data = []
    counter = 0

    for idx, row in tqdm(prompts.iterrows(), total=len(prompts), desc="Processing Prompts"):
        task = row['prompt']
        counter += 1

        wide_data_point = {
            "rational_last2": None,
            "emotional_last2": None,
            "rational_last": None,
            "emotional_last": None,
            "ballot_loop_merge_idx": counter,
            "ballot_no": row['ballot_no'],
        }

        for condition, prompt in CONDITIONS.items():

            a1 = Agent(model=random.choice(MODELS))
            a2 = Agent(model=random.choice(MODELS))
            debate = Debate(agents=[a1, a2], task=task, combination_instructions=prompt, cycles=2)
            debate.process()

            final_response = debate.final_response
            clean_final_response = final_response.replace("[Debater 2]", "").replace("\n", "")

            long_data_point = {
                "task": task,
                "m1": a1.model,
                "m2": a2.model,
                "raw_final": final_response,
                "clean_final": clean_final_response,
                "responses": debate.responses,
                "condition": condition,
                "last_two": last_two_responses(debate.responses),
                "ballot_loop_merge_idx": counter,
                "ballot_no": row['ballot_no'],
            }

            wide_data_point[f"{condition}_last2"] = last_two_responses(debate.responses)
            wide_data_point[f"{condition}_last"] = clean_response_for_qualtrics(clean_final_response)

            long_data.append(long_data_point)

        wide_data.append(wide_data_point)

    long_df = pd.DataFrame(long_data)
    long_df.to_json("../data/raw/ballot_deliberation_responses_long.jsonl", orient="records", lines=True)

    wide_df = pd.DataFrame(wide_data)
    wide_df.to_csv("../data/raw/ballot_deliberation_responses_wide.csv", index=False)

    for idx, row in wide_df.iterrows():
        print(f"Ballot {row['ballot_no']}")
        print("="*20)
        pprint(f"Emotional: {row['emotional_last']}")
        print("="*20)
        pprint(f"Rational: {row['rational_last']}")
        print("="*20)
        print()



if __name__ == "__main__":
    main()
