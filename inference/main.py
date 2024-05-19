from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

class Message(BaseModel):
    essay: str
    instruction: str

# Getting the prompt from the prompt.txt file
prompt_dir = "prompt.txt"
prompt = ''
with open(prompt_dir, 'r') as file:
    prompt = file.read()


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

def pre_process(instruction, essay):
    text = f"Instruction:{instruction}\nEssay:{essay}"
    return text

def generate_prompt(input):
    text = f"""{prompt}\n{input}"""
    return text

# Initialize your pipeline outside of your route functions
pipe = pipeline(
    "text-generation", 
    model="gildead/mistral-aes-414",
    device_map="auto"
)

# Initialize your FastAPI application
app = FastAPI()

# Define your route functions
@app.get("/")
async def root():
    return {"message": "Mistral API is running."}

@app.post("/score")
async def overall(message: Message):
    text = pre_process(message.instruction, message.essay)
    prompt = generate_prompt(text)

    result = pipe(
        f"<s>[INST] {prompt} [/INST]",
        max_new_tokens=200, 
        num_return_sequences=1,)

    generated_text = result[0]['generated_text']
    output = generated_text.split('[/INST]', 1)[-1].strip()
    final_output = post_process(output)

    return {"result": final_output}
