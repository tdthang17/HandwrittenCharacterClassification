import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình đã huấn luyện
model = load_model('model/model.keras')

# Gán nhãn
labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img_path = filepath

            # Tiền xử lý ảnh cho VGG19
            img = Image.open(filepath).convert('RGB')  # Chuyển ảnh sang RGB
            img = img.resize((224, 224))  # Resize đúng kích thước
            img = np.array(img) / 255.0  # Normalize
            img = img.reshape(1, 224, 224, 3)  # Định dạng đúng cho VGG19

            # Dự đoán
            pred = model.predict(img)
            pred_class = np.argmax(pred)
            prediction = labels[pred_class]

    return render_template('index.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)