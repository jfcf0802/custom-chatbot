from flask import Flask, request, jsonify, render_template, session
from model_2 import get_conversation_response, summarize_text, allowed_file
from werkzeug.utils import secure_filename
import PyPDF2
import os

app = Flask(__name__)
conversation_history = None

# Set a secret key for session management
app.config['SECRET_KEY'] = 'cenas_key'

# Set upload folder and allowed file types
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to render the chat interface
@app.route("/")
def home():
    return render_template("index_2.html")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    data = request.json
    user_input = data.get("message", "")  # Ensure the key exists

    if not user_input:
        return jsonify({"response": "Please provide a message."})

    response, conversation_history = get_conversation_response(user_input, conversation_history)

    return jsonify({"response": response})

# Route for file upload and summarization
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from PDF
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        # Store the document text in the session
        session['document_text'] = text

        # Summarize the document
        summary = summarize_text(text)

        return jsonify({'summary': summary}), 200

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False)
