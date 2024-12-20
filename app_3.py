from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from model_3 import (
    initialize_model,
    process_uploaded_file,
    generate_response,
)

os.environ["HF_HOME"] = r"H:/Python tests/huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"H:/Python tests/huggingface"

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model and tokenizer
tokenizer, model = initialize_model()

# In-memory storage for uploaded file content
file_context = ""

@app.route("/")
def home():
    return render_template("index_3.html")

@app.route("/chat", methods=["POST"])
def chat():
    global file_context
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "I didn't catch that. Could you say it again?"})

    # Generate response using file context if available
    response_text = generate_response(user_message, file_context, tokenizer, model)
    return jsonify({"response": response_text})

@app.route("/upload", methods=["POST"])
def upload_file():
    global file_context

    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file!"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process uploaded file and update the context
    try:
        file_context = process_uploaded_file(filepath, filename)
    except Exception as e:
        return jsonify({"message": f"Error processing file: {str(e)}"}), 500

    return jsonify({"message": "File uploaded and processed successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
