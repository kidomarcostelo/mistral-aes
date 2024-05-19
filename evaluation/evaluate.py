from nltk.translate import meteor
from nltk import word_tokenize
import nltk
import pandas as pd

import json
import requests

import os

from bert_score import BERTScorer

# nltk.download('wordnet')
# nltk.download('punkt')

dir="postprocess_files"
tosavedir="postprocess_eval_results_test_2\\test_data"

#toload = f'{dir}post_processed_eval_results_v2_414_validate_dataset.csv'
#tosave = f'{tosavedir}MistralV2_Results_414.csv'

toload = os.path.join(dir, 'eval-metrics-v2-414.csv')
tosave = os.path.join(tosavedir, 'MistralV2_Results_414_cl.csv')
 
# Load the csv file
df = pd.read_csv(toload)

# Initialize the sentence transformer model for STS
scorer = BERTScorer(lang='en')

api_token = "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"
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Initialize lists to store the scores
meteor_scores = []
sts_scores = []
bert_scores = []
bert_precision = []
bert_recall = []
bert_f1 = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # METEOR
    pred = word_tokenize(row['Expected Output'])
    ref = word_tokenize(row['Actual Output'])

    mscore = meteor([pred], ref)
    meteor_scores.append(mscore)

    # STS
    data = query(
    {
        "inputs": {
            "source_sentence": row['Expected Output'],
            "sentences":[row['Actual Output']]
        }
    })
    sts_scores.append(data[0])

    # BERTScore    
    # TODO - Do not round the scores
    P, R, F1 = scorer.score([row['Expected Output']], [row['Actual Output']])
    #bert_scores.append("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(P.item(), R.item(), F1.item()))

    bert_precision.append(P.item())
    bert_recall.append(R.item())
    bert_f1.append(F1.item())


# Add the scores to the dataframe
df['Meteor'] = meteor_scores
df['STS'] = sts_scores
df['BERT Precision'] = bert_precision
df['BERT Recall'] = bert_recall
df['BERT F1'] = bert_f1

# Calculate the overall score as the average of the three scores

# Save the dataframe to a new csv file
df.to_csv(tosave, index=False)

print("The scores have been successfully calculated and saved to output_with_scores.csv")
