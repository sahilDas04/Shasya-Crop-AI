from flask import Flask, request, jsonify
import PyPDF2
import io

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract_pdf_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)