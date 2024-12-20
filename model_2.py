from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from werkzeug.utils import secure_filename

# Set up the environment variable for HuggingFace and initialize the desired model.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load HuggingFace pipelines and models
conversation_model_name =  "meta-llama/Llama-3.2-3B-Instruct" 
summarizer_model_name = 'facebook/bart-large-cnn'
# qa_model_name = "deepset/roberta-base-squad2"

conversation_tokenizer = AutoTokenizer.from_pretrained(conversation_model_name, token=HF_TOKEN)
conversation_model = AutoModelForCausalLM.from_pretrained(conversation_model_name, token=HF_TOKEN)

summarizer = pipeline('summarization', model=summarizer_model_name)
# qa_pipeline = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name)

def get_conversation_response(prompt, history):
    if history==None:
        prompt = (
            f"### Instructions ###\n"
            f"You are a helpful assistant called The Smarty One (unless otherwise stated by the user). Answer the user's question or respond to their statement: {prompt}.\n"
            f"Start your response with 'Assistant: ' without restating the instructions.\n"
        )
    
    # Encode the input with an attention mask
    input_ids = conversation_tokenizer.encode(prompt + conversation_tokenizer.eos_token, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    # Combine with history if available
    if history is not None:
        bot_input_ids = torch.cat([history, input_ids], dim=-1)
        attention_mask = torch.cat([torch.ones_like(history), attention_mask], dim=-1)
    else:
        bot_input_ids = input_ids

    # Generate the response
    response = conversation_model.generate(
        bot_input_ids, 
        attention_mask=attention_mask, 
        max_length=1000, 
        pad_token_id=conversation_tokenizer.eos_token_id
    )
    
    response = conversation_tokenizer.decode(response[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Decode and return the response and history
    return response.split("Assistant: ")[-1].strip(), bot_input_ids

# Summarize the document with error handling
def summarize_text(text, max_length=1000, min_length=50):
    # Ensure the input text is non-empty
    if not text.strip():
        return "The document is empty or contains unreadable content."
    
    # HuggingFace models usually accept up to 1024 tokens; truncate if necessary
    token_limit = 1024  # Adjust based on your model's limit
    truncated_text = text[:token_limit]  # Truncate the text

    try:
        # Use the summarization model
        summary_result = summarizer(truncated_text, max_length=max_length, min_length=min_length, do_sample=False)

        # Check if the summarizer returned results
        if summary_result and len(summary_result) > 0:
            return summary_result[0]['summary_text']
        else:
            return "The summarization model did not return any results."
    except Exception as e:
        # Handle unexpected errors
        return f"An error occurred during summarization: {str(e)}"
    
# Helper function to check file type
def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS