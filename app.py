import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình đã huấn luyện
model = load_model('model/model.keras')

# Gán nhãn
labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def preprocess_image(img):
    # Chuyển đổi ảnh sang RGB và resize
    img = img.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files and 'drawing' not in request.form:
        return jsonify({'error': 'No image provided'}), 400

    img = None
    img_path = None

    # Xử lý upload file
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = Image.open(filepath)
            img_path = filepath

    # Xử lý hình vẽ từ canvas
    elif 'drawing' in request.form:
        img_data = request.form['drawing'].split(',')[1]  # Bỏ phần header base64
        img = Image.open(io.BytesIO(base64.b64decode(img_data)))
        # Tạo nền trắng cho hình vẽ
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
        img = background

    if img:
        # Tiền xử lý ảnh
        img_array = preprocess_image(img)

        # Dự đoán
        pred = model.predict(img_array)
        pred_class = np.argmax(pred)
        prediction = labels[pred_class]
        confidence = float(np.max(pred))

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'img_path': img_path.replace('\\', '/') if img_path else None
        })

    return jsonify({'error': 'Could not process image'}), 400


if __name__ == '__main__':
    app.run(debug=True)