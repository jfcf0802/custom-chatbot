# Custom Chatbot

This project is web application built using [Flask](https://flask.palletsprojects.com/en/stable/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) to create a personalized chatbot that also accepts documents as inputs. The application uses the [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model for natural language processing tasks.

## Features

- A conversational chatbot which allows for:
  - Conversations
  - Question/Answering'
  - File upload an analysis

## How It Works

1. **Flask Interface**:
   - A web-based interface is built with Flask, allowing users to have a conversation with the chatbot.
   - The Flask interface also allows for a high degree of personalization.

2. **Conversation Pipeline**:
   - A pipeline is set up to provide answers based on the user's inputs, always storing everything in the model's conversation history.
   - The model is prompted with user-provided data to create personalized answers.

1. **Model and Tokenizer Initialization**:
   - The application uses the `meta-llama/Llama-3.2-3B-Instruct` model from Hugging Face for text generation.
   - The tokenizer and model are loaded using the `transformers` library.

## Requirements

- Python 3.12
- Required Python packages:
  - `flask`
  - `transformers`
  - `torch`
  - `PyPDF2`
- Setup HuggingFace API key and Flask personalized key on CMD (`set HUGGINGFACEHUB_API_TOKEN=your_api_key` and `set APP_KEY=your_key`)

## Usage

1. Run the Python application on CMD
1. Open the application in your browser.
2. Input questions (or other conversational topics) and/or upload a pdf file.
3. Click the "Send" button or press ENTER
4. The personalized answers will appear as a conversation with the chatbot.


## Example


