import re
import unicodedata
import pandas as pd
from plurals.agent import Agent
from plurals.deliberation import Graph
import html
from dotenv import load_dotenv
import os
from pprint import pprint
from tqdm import tqdm

load_dotenv(dotenv_path="../src/.env")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY", "")


# Parameters
#####################
N_WORDS = 50
MODEL = "claude-3-sonnet-20240229"
N = 15
#####################

COT_PROMPT = f"""INSTRUCTIONS 
Generate a realistic description of a charter school that a liberal with a child would send their kids to. 

Follow the following format:

Rationale: In order to $produce the Description, we...
Description: A {N_WORDS}-word description of a charter school
"""

REVISE_PROMPT = f"""INSTRUCTIONS 
Generate a realistic description of a charter school that a liberal with a child would send their kids to. 

Follow the following format:

Rationale: In order to $produce the Description, and carefully and thoughtfully taking into account previous critiques, we...
Description: A {N_WORDS}-word description of a charter school
"""

critique_prompt = """INSTRUCTIONS
Given a description of a charter school, offer specific critiques for why you would not want to send your kid to this charter school. Be specific. You are in a focus group.

Critique: 
"""


def clean_text(text):
    # text = html.escape(text, quote=False)

    replacements = {
        '—': '&mdash;',  # em dash
        '–': '&ndash;',  # en dash
        '"': '&ldquo;',  # left double quotation mark
        '"': '&rdquo;',  # right double quotation mark
        ''': '&lsquo;',  # left single quotation mark
        ''': '&rsquo;',  # right single quotation mark
        '…': '&hellip;', # ellipsis
    }

    for char, entity in replacements.items():
        text = text.replace(char, entity)

    return text


def main():


    structure_data = []
    data = []
    for i in tqdm(range(N)):
        zero_shot = Agent(model=MODEL, task=COT_PROMPT).process()

        agents = {
            "init_arguer": Agent(task=COT_PROMPT,
                                 model=MODEL),
            "critic_1": Agent(
                query_str="ideo5=='Liberal'&child18=='Yes'",
                task=critique_prompt,
                model=MODEL,
                combination_instructions="Use previous responses: ${previous_responses}",
            ),
            "critic_2": Agent(
                query_str="ideo5=='Liberal'&child18=='Yes'",
                task=critique_prompt,
                model=MODEL,
                combination_instructions="Use previous responses: ${previous_responses}",
            ),
            "critic_3": Agent(
                query_str="ideo5=='Liberal'&child18=='Yes'",
                task=critique_prompt,
                model=MODEL,
                combination_instructions="Use previous responses: ${previous_responses}",
            ),
            "final_arguer": Agent(
                task=REVISE_PROMPT,
                model=MODEL,
                combination_instructions="Use previous critiques for a description: ${previous_responses}",
            ),
        }

        edges = [
            ("init_arguer", "critic_1"),
            ("init_arguer", "critic_2"),
            ("init_arguer", "critic_3"),
            ("critic_1", "final_arguer"),
            ("critic_2", "final_arguer"),
            ("critic_3", "final_arguer")

        ]

        graph = Graph(agents, edges)
        graph.process()
        extract_argument = lambda response: clean_text(response.split("Description:")[-1].strip())

        data_pt = {
            "zero_shot": extract_argument(zero_shot),
            "final": extract_argument(graph.responses[-1]),
            'idx':i+1
        }
        pprint(data_pt)
        data.append(data_pt)
        structure_info = graph.info
        structure_info.update({'idx':i+1})
        structure_data.append(structure_info)


    df = pd.DataFrame(data)
    df.to_csv(f"../data/raw/charter_{MODEL}.csv", index=False)

    structure_df = pd.DataFrame(structure_data)
    structure_df.to_json(f"../data/raw/charter_structure_{MODEL}.jsonl", orient='records', lines=True)

if __name__ == "__main__":
    main()