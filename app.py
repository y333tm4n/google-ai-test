from dotenv import load_dotenv
import os
from google import genai
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

pdf_path = input("Enter the path to your PDF file: ")
document_text = extract_text_from_pdf(pdf_path)

# Summarize the document
summary_prompt = f"Summarize the following document in a concise way:\n\n{document_text}"
summary_response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=summary_prompt,
)
print("\n--- Document Summary ---")
print(summary_response.text)

# Question answering loop
while True:
    user_question = input("\nAsk a question about the document (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        break
    qa_prompt = f"Based on the following document, answer the question:\n\nDocument:\n{document_text}\n\nQuestion: {user_question}"
    qa_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=qa_prompt,
    )
    print("Answer:", qa_response.text)