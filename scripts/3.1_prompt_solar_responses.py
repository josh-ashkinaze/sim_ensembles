import re
import unicodedata
import pandas as pd
from tqdm import tqdm
from plurals.agent import Agent
from plurals.deliberation import Moderator, Ensemble
from pprint import pprint
import ftfy
import html
import argparse


def pretty_print_df(df):
    """
    Print a formatted version of the DataFrame for easy reading.
    """
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
    """
    Clean and normalize the input text, replacing special characters with HTML entities.
    """
    text = strip_response_string(text)
    text = unicodedata.normalize('NFKC', text)
    text = html.escape(text, quote=False)

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


def main(n, model, temp):
    """
    Main function to generate and process responses for solar panel products.

    Args:
    n (int): Number of iterations
    model (str): Model name to use for generating responses
    """
    if temp:
        kwargs = {'temperature': temp}
    else:
        kwargs = {}

    data_pts = []
    for i in tqdm(range(n), desc="Processing prompts"):
        # Focus group task and participants
        focus_group_task = "What specific product details for a solar panel company would resonate with you personally? Be very specific; you are in a focus group. Answer in 20 words."
        focus_group_participants = [Agent(model=model, kwargs=kwargs, task=focus_group_task, ideology='conservative')
        for _ in
                                    range(10)]

        moderator = Moderator(model=model,
                              kwargs=kwargs,
                              system_instructions="You are an expert copywriter for an ad agency.",
                              task="You are overseeing a focus group discussing what products would resonate with them for the solar panel category.",
                              combination_instructions=f"Here are focus group responses: \n<start>${{previous_responses}}<end>. Now based on the specifics of these responses, come up with a specific product for a solar panel company that would resonate with the focus group members. Be very specific. Answer in 50 words only.")

        ensemble = Ensemble(agents=focus_group_participants, moderator=moderator)
        ensemble.process()
        print(ensemble.moderator.history)

        ensemble_response = ensemble.final_response

        # Default
        zero_shot_task = "Come up with a specific product for a solar panel company that would resonate with conservatives. Be very specific. Answer in 50 words only."
        zero_shot = Agent(model=model,
                          kwargs=kwargs,
                          system_instructions="You are an expert copywriter for an ad agency.",
                          task=zero_shot_task)
        zero_shot_response = zero_shot.process()

        data_pt = {'focus_group_final': ensemble_response, 'zero_shot': zero_shot_response, 'idx': i + 1,
                   'focus_group_all': ensemble.responses}
        data_pts.append(data_pt)

    df = pd.DataFrame(data_pts)
    df['clean_focus_group_final'] = df['focus_group_final'].apply(clean_text)
    df['clean_zero'] = df['zero_shot'].apply(clean_text)

    df.to_json(f"../data/raw/solar_panels_conservatives_{model}_{n}.jsonl", orient='records', lines=True)
    df.drop(columns=['focus_group_all']).to_csv(f"../data/raw/solar_panels_conservatives_{model}_{n}.csv", index=False)

    print("Data processing complete.")

    pretty_print_df(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and process responses for solar panel products.")
    parser.add_argument("n", type=int, help="Number of iterations")
    parser.add_argument("model", type=str, help="Model name to use for generating responses")
    parser.add_argument("--temp", type=float, help="Temperature setting for the model (optional)")

    args = parser.parse_args()

    main(args.N, args.MODEL, args.TEMP)