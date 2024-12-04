from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Set up the environment variable for HuggingFace and initialize the desired model.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

conversation_model_name =  "meta-llama/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.1-8B-Instruct" # "meta-llama/Meta-Llama-3-8B" # "microsoft/DialoGPT-medium" # "bigscience/bloom" # 
conversation_tokenizer = AutoTokenizer.from_pretrained(conversation_model_name, token=HF_TOKEN)
conversation_model = AutoModelForCausalLM.from_pretrained(conversation_model_name, token=HF_TOKEN)

def get_conversation_response(prompt, history):
    if history==None:
        prompt = f"You are a helpful assistant. Answer the user's question or respond to their statement.\n\nUser: {prompt}\nAssistant:"
    
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
    
    print(f'response 1: {response}')
    
    response = conversation_tokenizer.decode(response[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    print(f'prompt: {prompt}')
    print(f'history: {history}')
    print(f'response 2: {response}')

    # Decode and return the response and history
    return response, bot_input_ids

# # Ensure tokenizer has a padding token
# if conversation_tokenizer.pad_token is None:
#     conversation_tokenizer.pad_token = conversation_tokenizer.eos_token

# def get_conversation_response(prompt, history=None):
#     """
#     Generate a conversational response from the model.
#     """
#     instruction = "You are a helpful assistant. Answer the user's question or respond to their statement."
#     full_prompt = f"{instruction}\n\nUser: {prompt}\nAssistant:"
#     print(f"Full Prompt: {full_prompt}")  # Debug

#     # Encode the input
#     input_ids = conversation_tokenizer.encode(full_prompt, return_tensors="pt")
#     attention_mask = torch.ones_like(input_ids)

#     # Combine input with history if available
#     if history is not None:
#         bot_input_ids = torch.cat([history, input_ids], dim=-1)
#         attention_mask = torch.cat([torch.ones_like(history), attention_mask], dim=-1)
#     else:
#         bot_input_ids = input_ids

#     try:
#         # Generate the response with tuned parameters
#         response_ids = conversation_model.generate(
#             bot_input_ids,
#             attention_mask=attention_mask,
#             max_length=100,
#             num_beams=5,
#             no_repeat_ngram_size=2,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.9,
#             pad_token_id=conversation_tokenizer.pad_token_id
#         )

#         # Decode the response
#         decoded_response = conversation_tokenizer.decode(
#             response_ids[:, bot_input_ids.shape[-1]:][0],
#             skip_special_tokens=True
#         )
#         print(f"Decoded Response: {decoded_response}")  # Debug

#         # Handle empty or unwanted responses
#         if not decoded_response.strip():
#             decoded_response = "I'm here to assist, but I couldn't understand that. Can you clarify?"
#         elif decoded_response.startswith("assistant"):
#             decoded_response = decoded_response[len("assistant"):].strip()

#         return decoded_response, bot_input_ids

#     except Exception as e:
#         print(f"Error during generation: {e}")
#         return "I encountered an error while processing your request. Please try again.", history

    
test_prompt = "Hello"
response, history = get_conversation_response(test_prompt, None)
print(f"Bot: {response}")