"""
Author: Joshua Ashkinaze

Description: Takes the raw charter school data from Qualtrics processes it into a clean format.
"""

import pandas as pd


def clean_choice(x):
    """
    Clean choice column.

    NOTE:
        - For the charter school data, field 2 is the focus group and field 1 is zero shot.
        - Order is always randomized when displayed to participants anyway though

    """
    if "Field/2" in x:
        return "Simulated Focus Group\n(Directed Acyclic Graph)"
    elif "Field/1" in x:
        return "Zero Shot"
    else:
        raise ValueError


def comp_correct(x):
    """Flags if got comprehension check correct"""
    if "Charter management organizations" in x:
        return 1
    else:
        return 0


def main():
    df = pd.read_csv('../data/raw/qualtrics_charter.csv')
    df = df.iloc[2:]

    data_pts = []
    for idx, row in df.iterrows():
        uid = row['ResponseId']
        duration = float(row['Duration (in seconds)']) / 60
        age = row['age']
        gender = row['gender']
        educ = row['educ']
        commit = row['commitment']
        comprehension = row['charter_school']

        for i in range(1, 16):
            data_pt = {'uid': uid, 'duration': duration, 'age': age, 'gender': gender, 'educ': educ, 'commit': commit,
                'choice': row[f'{i}_lm'], 'comprehension': comprehension

            }
            data_pts.append(data_pt)

    df = pd.DataFrame(data_pts)
    df = df.dropna()
    df['clean_choice'] = df['choice'].apply(clean_choice)
    df['focus_chosen'] = df['clean_choice'].apply(lambda x: 1 if "Simulated Focus Group" in x else 0)
    df['comp_correct'] = df['comprehension'].apply(comp_correct)

    # assertions
    assert df['uid'].nunique() == 20, "not 20 ppl"
    assert df['clean_choice'].nunique() == 2, "not 2 choices"
    assert len(df) == 300, "not 300 data pts"

    df.to_csv('../data/processed/clean_school_data.csv', index=False)
    print("success")


if __name__ == '__main__':
    main()
