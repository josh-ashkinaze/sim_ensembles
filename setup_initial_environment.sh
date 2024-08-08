#!/bin/bash

# create venv
python3 -m venv venvt
source venv/bin/activate

# install common packages I use
pip install litellm requests beautifulsoup4 seaborn matplotlib scikit-learn pandas openai sentence-transformers scipy
 statsmodels plurals

# freeze versions right now
pip freeze > requirements.txt

echo "Setup complete and requirements saved."
