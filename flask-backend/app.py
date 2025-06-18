from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from test import run_model  # Import the callable function

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

        # Save image
        os.makedirs("temp_images", exist_ok=True)
        image_path = os.path.join("temp_images", image_name)

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_data.split(",")[1]))

        # Run model and get report
        report = run_model(image_path, patient_id=patient_id)

        return jsonify({"output": report})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
