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
N_WORDS = 75
MODEL = "claude-3-sonnet-20240229"
N = 15
#####################

COT_PROMPT = f"""INSTRUCTIONS 
Produce a compelling proposal for a homeless shelter addressed to local residents who are liberals. Give specific details.

Follow the following format:

Rationale: In order to produce a compelling $Proposal, we...
Proposal: A {N_WORDS}-word proposal addressed to residents, starting with "Dear residents, ..."

Constraints:
- Do not add placeholders like [details]
"""

REVISE_PROMPT = f"""INSTRUCTIONS 
Produce a compelling proposal for a homeless shelter addressed to local residents who are liberals. Give specific details.

Follow the following format:
    
Rationale: In order to produce a compelling $Proposal, and carefully and thoughtfully taking into account previous critiques from residents, we...
Proposal: A {N_WORDS}-word proposal addressed to residents, starting with "Dear residents, ..."

Constraints:
- Do not add placeholders like [details]
"""

feedback_prompt = """INSTRUCTIONS
Given a proposal for a homeless shelter, offer feedback that would make you more likely to accept this proposal. Be specific. You are in a focus group.

Critique: 
"""


def clean_text(text):
    # sometimes llm added things like "***" at the start text
    text = re.sub(r'^\W+', '', text)


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
                query_str="ideo5=='Liberal'",
                task=feedback_prompt,
                model=MODEL,
                combination_instructions="default",
            ),
            "critic_2": Agent(
                query_str="ideo5=='Liberal'",
                task=feedback_prompt,
                model=MODEL,
                combination_instructions="default",
            ),
            "critic_3": Agent(
                query_str="ideo5=='Liberal'",
                task=feedback_prompt,
                model=MODEL,
                combination_instructions="default",
            ),
            "final_arguer": Agent(
                task=REVISE_PROMPT,
                model=MODEL,
                combination_instructions="default",
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
        extract_argument = lambda response: clean_text(response.split("Proposal:")[-1].strip())

        data_pt = {
            "zero_shot": extract_argument(zero_shot),
            "final": extract_argument(graph.responses[-1]),
            'idx':i+1
        }
        pprint(data_pt)
        pprint(agents['critic_1'].responses)
        data.append(data_pt)
        structure_info = graph.info
        structure_info.update({'idx':i+1})
        structure_data.append(structure_info)


    df = pd.DataFrame(data)
    df.to_csv(f"../data/raw/house_{MODEL}_{N_WORDS}.csv", index=False)

    structure_df = pd.DataFrame(structure_data)
    structure_df.to_json(f"../data/raw/house_structure_{MODEL}_{N_WORDS}.jsonl", orient='records', lines=True)

if __name__ == "__main__":
    main()