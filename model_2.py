from transformers import pipeline
import os

# Set up the HuggingFace token and model
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the model name
conversation_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your desired model

# Set up the conversational pipeline
chat_pipeline = pipeline(
    "text-generation",
    model=conversation_model_name,
    tokenizer=conversation_model_name,
    token=HF_TOKEN
)

def get_conversation_response(prompt, history=None):
    """
    Generate a conversational response using the HuggingFace pipeline.
    Args:
        prompt (str): The user input.
        history (str): The conversation history (if any).
    Returns:
        str, str: The bot response and updated history.
    """
    if history is None:
        history = ""

    # Combine the history and the user prompt
    input_text = history + f"User: {prompt}\nBot:"

    # Generate a response using the pipeline
    response = chat_pipeline(input_text, max_length=200, pad_token_id=chat_pipeline.tokenizer.eos_token_id)

    # Extract the generated response
    bot_response = response[0]["generated_text"].split("Bot:")[-1].strip()

    # Update the history
    updated_history = input_text + f" {bot_response}\n"

    return bot_response, updated_history
