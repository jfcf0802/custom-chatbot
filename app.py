from flask import Flask, request, jsonify, render_template, session
from model import get_conversation_response, allowed_file, extract_summarize_text
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
conversation_history = None

# Set a secret key for session management
app.config['SECRET_KEY'] = os.getenv("APP_KEY")

# Set upload folder and allowed file types
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to render the chat interface
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    data = request.json
    user_input = data.get("message", "")  # Ensure the key exists

    if not user_input:
        return jsonify({"response": "Please provide a message."})

    response, conversation_history = get_conversation_response(
        user_input, conversation_history)

    return jsonify({"response": response})

# Route for file upload and summarization
@app.route('/upload', methods=['POST'])
def upload_file():
    global conversation_history

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text, summary, conversation_history = extract_summarize_text(
            file_path, conversation_history)

        # Store the document text in the session
        session['document_text'] = text

        return jsonify({'summary': summary}), 200

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False)
