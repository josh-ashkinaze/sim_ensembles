"""
Author: Joshua Ashkinaze

Description: Takes the raw solar panel data from Qualtrics processes it into a clean format.
"""

import pandas as pd


def clean_choice(x):
    """Clean the choice column"""
    if "Field/1" in x:
        return "Simulated Focus Group\n(Moderated Ensemble)"
    elif "Field/2" in x:
        return "Zero Shot"
    else:
        raise ValueError


def main():
    df = pd.read_csv('../data/raw/qualtrics_solar_panel1.csv')
    df = df.iloc[2:]

    data_pts = []
    for idx, row in df.iterrows():
        uid = row['ResponseId']
        duration = float(row['Duration (in seconds)']) / 60
        age = row['age']
        gender = row['gender']
        educ = row['educ']
        commit = row['commitment']

        for i in range(1, 16):
            data_pt = {'uid': uid, 'duration': duration, 'age': age, 'gender': gender, 'educ': educ, 'commit': commit,
                       'choice': row[f'{i}_lm']

                       }
            data_pts.append(data_pt)

    df = pd.DataFrame(data_pts)
    df = df.dropna()
    df['clean_choice'] = df['choice'].apply(clean_choice)
    df['focus_chosen'] = df['clean_choice'].apply(lambda x: 1 if "Simulated Focus Group" in x else 0)

    # assertions
    assert df['uid'].nunique() == 20, "not 20 ppl"
    assert df['clean_choice'].nunique() == 2, "not 2 choices"
    assert len(df) == 300, "not 300 data pts"

    df.to_csv('../data/processed/clean_solar_data.csv', index=False)
    print("success")


if __name__ == '__main__':
    main()
