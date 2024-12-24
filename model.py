from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import PyPDF2

# Set up the environment variable for HuggingFace and initialize the desired model.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load HuggingFace pipelines and models
conversation_model_name =  "meta-llama/Llama-3.2-3B-Instruct" 

conversation_tokenizer = AutoTokenizer.from_pretrained(conversation_model_name, token=HF_TOKEN)
conversation_model = AutoModelForCausalLM.from_pretrained(conversation_model_name, token=HF_TOKEN)

def get_conversation_response(prompt, history, max_length=2024):
    if history==None:
        prompt = (
            f"### Instructions ###\n"
            f"You are a helpful assistant called The Smarty One (unless otherwise stated by the user). Answer the user's question or respond to their statement: {prompt}.\n"
            f"Start your response with 'Assistant: ' without restating the instructions.\n"
        )
    
    # Encode the input with an attention mask
    input_ids = conversation_tokenizer.encode(prompt + conversation_tokenizer.eos_token, return_tensors="pt")

    # Combine with history if available
    if history is not None:
        bot_input_ids = torch.cat([history, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids
        
    attention_mask = torch.ones_like(bot_input_ids)

    # Generate the response
    response = conversation_model.generate(
        bot_input_ids, 
        attention_mask=attention_mask, 
        max_length=max_length, 
        pad_token_id=conversation_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
    
    response = conversation_tokenizer.decode(response[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Decode and return the response and history
    return response.split("Assistant: ")[-1].strip(), bot_input_ids
    
# Extract text from pdf
def extract_summarize_text(file_path, history, max_length=2024, limit_file_text=1024):
    # Extract text from PDF
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    if not False:
        text = text[:limit_file_text]
    
    if history==None:
        prompt = (
            f"### Instructions ###\n"
            f"You are a helpful assistant called The Smarty One (unless otherwise stated by the user).\n"
            f"The user uploaded a document containing: {text}.\n"
            f"Summarize the document's content and answer the user's question or respond to their statement\n"
            f"Start your response with 'Assistant: ' without restating the instructions.\n"
        )
    else:
        prompt = (
            f"### Instructions ###\n"
            f"The user uploaded a document containing: {text}.\n"
            f"Summarize the document's content and answer the user's question or respond to their statement\n"
            f"Start your response with 'Assistant: ' without restating the instructions.\n"
        )
        
    # Encode the input with an attention mask
    input_ids = conversation_tokenizer.encode(prompt + conversation_tokenizer.eos_token, return_tensors="pt")
        
    # Combine with history if available
    if history is not None:
        bot_input_ids = torch.cat([history, input_ids], dim=-1)
        attention_mask = torch.ones_like(bot_input_ids)
    else:
        bot_input_ids = input_ids
        attention_mask = torch.ones_like(bot_input_ids)
    
    # Generate the response
    response = conversation_model.generate(
        bot_input_ids, 
        attention_mask=attention_mask, 
        max_length=max_length, 
        pad_token_id=conversation_tokenizer.eos_token_id
    )
    
    response = conversation_tokenizer.decode(response[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Decode and return the response and history
    summary = response.split("Assistant: ")[-1].strip()
    
    return(text, summary, bot_input_ids)
    
# Helper function to check file type
def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS