from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import base64
from test import run_model

app = Flask(__name__)
CORS(app)

@app.route('/run-test', methods=['POST'])
def run_test():
    try:
        data = request.json
        image_data = data.get('imageData')
        image_name = data.get('imageName')
        patient_id = data.get('patientId', 'Unknown')

        if not image_data or not image_name:
            return jsonify({'error': 'Missing image data or image name'}), 400

        # Save the image to temp_images
        os.makedirs("temp_images", exist_ok=True)
        image_path = os.path.join("temp_images", image_name)

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_data.split(",")[1]))

        # Run the AI model and get the report + pdf path
        result = run_model(image_path, patient_id=patient_id)

        return jsonify({
            "output": result["text"],
            "pdfUrl": f"http://localhost:5000/get-pdf/{patient_id}",
            "timestamp": result.get("timestamp", "")
})


    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-pdf/<patient_id>', methods=['GET'])
def get_pdf(patient_id):
    pdf_path = f"temp_images/{patient_id}_report.pdf"
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype="application/pdf")
    return "PDF not found", 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
