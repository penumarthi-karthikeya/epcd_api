from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from PIL import Image
import numpy as np
import io
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'message': 'API is up and running!'}), 200

class Predict(Resource):
    def post(self):
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400
        
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            
            # Use the TFLite model to make a prediction
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            # Assuming binary classification
            class_labels = ['normal', 'pancreatic_tumor']
            predicted_class = (prediction > 0.5).astype(int)
            predicted_label = class_labels[predicted_class[0][0]]
            
            return {'predicted_class': predicted_label}
        except Exception as e:
            return {'error': str(e)}, 500
    
    def get(self):
        return {'error': 'GET method not allowed. Use POST method instead.'}, 405

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run()