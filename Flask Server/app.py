from flask import Flask, request, jsonify
import PyPDF2
import io
import re

app = Flask(__name__)

def extract_features(text):
    features = {}
    patterns = {
        'nitrogen': r'Nitrogen[:\-]?\s*([\d\.]+)',
        'phosphorus': r'Phosphorus[:\-]?\s*([\d\.]+)',
        'potassium': r'Potassium[:\-]?\s*([\d\.]+)',
        'temperature': r'Temperature[:\-]?\s*([\d\.]+)',
        'humidity': r'Humidity[:\-]?\s*([\d\.]+)',
        'ph': r'Ph[:\-]?\s*([\d\.]+)',
        'rainfall': r'rain Fall[:\-]?\s*([\d\.]+)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        features[key] = float(match.group(1)) if match else None
    return features

@app.route('/extract', methods=['POST'])
def extract_pdf_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() or ''
    features = extract_features(text)
    return jsonify(features)

if __name__ == '__main__':
    app.run(debug=True)