from nltk.translate import meteor
from nltk import word_tokenize
import nltk
import pandas as pd


nltk.download('wordnet')
nltk.download('punkt')

predictions = word_tokenize("It is a guide to action which ensures that the military always obeys the commands of the party")
references = word_tokenize("It is a guide to action that ensures that the military will forever heed Party commands")
#predictions = "It is a guide to action which ensures that the military always obeys the commands of the party"
#references = "It is a guide to action that ensures that the military will forever heed Party commands"


meteor_score1 = meteor([predictions], references)


print(meteor_score1)


df = pd.read_csv('eval-metrics-v2-414.csv')

for index, row in df.iterrows():

    pred = word_tokenize(row['Expected Output'])
    ref = word_tokenize(row['Actual Output'])


    print(meteor([pred], ref))



import json
import requests

api_token = "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"
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

df = pd.read_csv('eval-metrics-v2-414.csv')

for index, row in df.iterrows():

    pred = word_tokenize(row['Expected Output'])
    ref = word_tokenize(row['Actual Output'])
    
    data = query(
    {
        "inputs": {
            "source_sentence": row['Expected Output'],
            "sentences":[row['Actual Output']]
        }
    })

    ## [0.605, 0.894]

    print(data)


from bert_score import BERTScorer

scorer = BERTScorer(lang='en')

# # Define the reference and candidate sentences
# reference = ["The cat sat on the mat."]
# candidate = ["A cat is sitting on a mat."]

# # Compute the BERTScore

print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(P.item(), R.item(), F1.item()))

df = pd.read_csv('eval-metrics-v2-414.csv')
for index, row in df.iterrows():

    pred = row['Expected Output']
    ref = row['Actual Output']
    
    P, R, F1 = scorer.score([pred], [ref])

    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(P.item(), R.item(), F1.item()))
