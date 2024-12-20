import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PyPDF2 import PdfReader
# import docx

# Set up the environment variable for HuggingFace and initialize the desired model.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def initialize_model():
    """Initialize the tokenizer and model."""
    model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with your desired model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
    return tokenizer, model

def generate_response(user_message, file_context, tokenizer, model):
    """Generate a response based on user input and file context."""
    if file_context:
        prompt = f"Context: {file_context}\nUser: {user_message}\nAssistant:"
    else:
        prompt = f"User: {user_message}\nAssistant:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    response_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Extract response text after "Assistant:"
    response_text = response.split("Assistant:")[-1].strip()
    return response_text

def process_uploaded_file(filepath, filename):
    """Process the uploaded file and extract its text content."""
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(filepath)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(filepath)
    elif filename.endswith('.txt'):
        with open(filepath, 'r', encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type!")

def extract_text_from_pdf(filepath):
    """Extract text from a PDF file."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(filepath):
    """Extract text from a Word document."""
    doc = docx.Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

