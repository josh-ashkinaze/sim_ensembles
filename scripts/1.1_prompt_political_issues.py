"""
Author: Joshua Ashkinaze

Description: Description

Date: 2024-08-27 12:47:04
"""



import os
import sys
import pandas as pd
from tqdm import tqdm
from plurals.agent import Agent
import dotenv
dotenv.load_dotenv("../src/.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from helpers import *
import lexical_diversity

def load_issues(file_path, n_issues, n_words):
    pol_issues = pd.read_csv(file_path)
    issues = pol_issues.sort_values(by=['initial_order'], ascending=True).head(n_issues)
    issues['description_instructions'] = issues['question'] + f" Answer in {n_words} words with a rationale."
    issues.to_csv("../data/processed/issue_prompts.csv")
    return issues[['issue', 'description_instructions']]

def create_agent(agent_type, ideology, model, used_personas, counter):
    kwargs = {'seed': counter} if "gpt-4" in model else {}

    if "persona_" in agent_type:
        while True:
            agent = Agent(ideology=ideology, model=model, kwargs=kwargs)
            if agent.persona not in used_personas:
                used_personas.add(agent.persona)
                return agent
    else:
        return Agent(system_instructions=f"You are a {ideology}", model=model,kwargs=kwargs)

def process_issues(issues, agent_types, model, n_per_block):
    data = []
    used_personas = set()
    total_iters = len(issues) * n_per_block * len(agent_types)

    with tqdm(total=total_iters) as pbar:
        counter = 0
        for _, row in issues.iterrows():
            topic = row['issue']
            prompt = row['description_instructions']
            for iter in range(n_per_block):
                for agent_type, agent_info in agent_types.items():
                    ideology = agent_info['ideology']
                    agent = create_agent(agent_type, ideology, model, used_personas, counter)
                    resp = agent.process(prompt)
                    data.append({
                        "topic": topic,
                        "agent_type": "persona" if "persona_" in agent_type else "default",
                        "ideology": ideology,
                        "prompt": prompt,
                        "response": resp,
                        "iter": iter,
                        "system_instructions": agent.system_instructions,
                    })
                    counter += 1
                    pbar.update(1)

    return pd.DataFrame(data)

def run_experiment(model, n_words, n_issues, n_per_block, input_file, output_file):
    issues = load_issues(input_file, n_issues, n_words)

    agent_types = {
        'persona_conservative': {"ideology": "conservative"},
        'persona_liberal': {"ideology": "liberal"},
        'default_conservative': {"ideology": "conservative"},
        'default_liberal': {"ideology": "liberal"},
    }

    results = process_issues(issues, agent_types, model, n_per_block)
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    models = ["gpt-4o", "claude-3-sonnet-20240229"]
    n_words = 100
    n_issues = 4
    n_per_block = 30
    input_file = "../data/raw/2024-08-08_isidewith_political_issues_popular.csv"

    for model in models:
        output_file = f"../data/processed/{n_words}words_{n_issues}issues_{n_per_block}iters_{model}_pol_responses.csv"
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping...")
        else:
            run_experiment(model, n_words, n_issues, n_per_block, input_file, output_file)

if __name__ == "__main__":
    main()