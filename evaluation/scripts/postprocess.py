import pandas as pd
import json

def post_process(essay):
    # Find the index of the first occurrence of the word "Feedback:"
    feedback_index = essay.find("Feedback:")

    # If "Feedback:" is not found, return the original essay
    if feedback_index == -1:
        return essay

    # Find the index of the newline after the first occurrence of "Feedback:"
    newline_index = essay.find("\n", feedback_index)

    # If newline is not found, return the original essay
    if newline_index == -1:
        return essay

    # Return the essay up to the newline after the first occurrence of "Feedback:"
    return essay[:newline_index]




df = pd.read_csv("eval-metrics-v2-966-validate_dataset.csv")
for index, row in df.iterrows():
    # Get the essay
    essay = row['Actual Output']

            # Post-process the essay
    processed_essay = post_process(essay)

# Update the dataframe with the processed essay
    df.at[index, 'Actual Output'] = processed_essay

# Save the updated dataframe to a new csv file
df.to_csv("post_processed_eval_results_v2_966_validate_dataset.csv", index=False)
