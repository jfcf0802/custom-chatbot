from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Set up the environment variable for HuggingFace and initialize the desired model.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

conversation_model_name =  "meta-llama/Llama-3.1-8B-Instruct" # "meta-llama/Meta-Llama-3-8B" # "microsoft/DialoGPT-medium" # "bigscience/bloom" # 
conversation_tokenizer = AutoTokenizer.from_pretrained(conversation_model_name, token=HF_TOKEN)
conversation_model = AutoModelForCausalLM.from_pretrained(conversation_model_name, token=HF_TOKEN)

def get_conversation_response(prompt, history):
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

    # Decode and return the response and history
    return conversation_tokenizer.decode(response[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), bot_input_ids

# def get_conversation_response(prompt, history):
#     # Encode the input with an attention mask
#     input_ids = conversation_tokenizer.encode(prompt + conversation_tokenizer.eos_token, return_tensors="pt")
#     attention_mask = torch.ones_like(input_ids)

#     # Combine with history if available
#     if history is not None:
#         bot_input_ids = torch.cat([history, input_ids], dim=-1)
#         attention_mask = torch.cat([torch.ones_like(history), attention_mask], dim=-1)
#     else:
#         bot_input_ids = input_ids

#     # Generate the response
#     response = conversation_model.generate(
#         bot_input_ids, 
#         attention_mask=attention_mask, 
#         max_length=1000, 
#         pad_token_id=conversation_tokenizer.eos_token_id
#     )

#     # Decode and return the response and history
#     return conversation_tokenizer.decode(response[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True), bot_input_ids

# print(get_conversation_response('Hello', None))

# def get_conversation_response(prompt, history):
#     # Add a conversation-specific prompt
#     conversation_prefix = (
#         "You are a helpful and friendly chatbot. Respond conversationally to the user's input.\n"
#         "User: Hello!\n"
#         "Bot: Hi there! How can I assist you today?\n"
#     )
    
#     # Append the user's prompt to the conversation
#     input_text = conversation_prefix
#     if history:
#         input_text += "".join(history)  # Combine past conversation if any
#     input_text += f"User: {prompt}\nBot:"

#     # Assign pad_token if not already defined
#     if conversation_tokenizer.pad_token is None:
#         conversation_tokenizer.pad_token = conversation_tokenizer.eos_token

#     # Define a maximum length for the input sequence
#     max_input_length = 1024  # You can adjust this value based on your model's limitations

#     # Encode the input and create the attention mask with truncation and padding
#     input_ids = conversation_tokenizer.encode(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)

#     # Create an attention mask (1 for valid tokens, 0 for padding tokens)
#     attention_mask = torch.ones_like(input_ids)  # By default, we assume all tokens are valid

#     # Generate the response
#     response = conversation_model.generate(
#         input_ids,
#         attention_mask=attention_mask,  # Pass the attention mask here
#         max_length=500,  # Set max length for output sequence
#         pad_token_id=conversation_tokenizer.pad_token_id,  # Use pad_token_id for padding
#         do_sample=True,  # Enable sampling for conversational variety
#         top_p=0.95,      # Use nucleus sampling for diverse output
#         temperature=0.7  # Adjust creativity of the response
#     )
    
#     # Decode and return the response
#     decoded_response = conversation_tokenizer.decode(response[0], skip_special_tokens=True)
    
#     # Extract only the bot's response (everything after 'Bot:')
#     bot_response = decoded_response.split("Bot:")[-1].strip()
    
#     # Update history for the next conversation round
#     history = [input_text + decoded_response]
    
#     return bot_response, history


