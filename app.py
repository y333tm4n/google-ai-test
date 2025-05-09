from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from google import genai
import PyPDF2
import tempfile

app = Flask(__name__)
CORS(app)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

def extract_text_from_pdf_file(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

@app.route("/summarize", methods=["POST"])
def summarize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        with open(tmp.name, "rb") as f:
            text = extract_text_from_pdf_file(f)
    summary_prompt = f"Summarize the following document in a concise way:\n\n{text}"
    summary_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=summary_prompt,
    )
    return jsonify({"summary": summary_response.text, "document_text": text})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.form
    document_text = data.get("document_text", "")
    question = data.get("question", "")
    if not document_text or not question:
        return jsonify({"error": "Missing document_text or question"}), 400
    qa_prompt = f"Based on the following document, answer the question:\n\nDocument:\n{document_text}\n\nQuestion: {question}"
    qa_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=qa_prompt,
    )
    return jsonify({"answer": qa_response.text})

if __name__ == "__main__":
    app.run(debug=True)