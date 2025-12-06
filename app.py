import os
import io
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
from tensorflow.keras.applications.vgg19 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import ImageOps, ImageFilter
import keras
from keras import layers, applications

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def recreate_model():
    # Download VGG19 as a base (but don't attach it to the input yet)
    # Don't use input_tensor here to ensure it is nested.
    base_model = applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze

    # Use Sequential to group
    model = keras.Sequential([
        keras.Input(shape=(224, 224, 3), name="input_layer_1"),

        # Stuff the entire VGG19 as a single layer
        base_model,
        layers.Flatten(name="flatten"),
        layers.Dense(224, activation="relu", name="dense"),
        layers.Dropout(0.1, name="dropout"),
        layers.Dense(416, activation="sigmoid", name="dense_1"),
        layers.Dropout(0.1, name="dropout_1"),
        layers.Dense(62, activation="softmax", name="dense_2")
    ])

    return model


model_path = "model/model.keras"
try:
    print("Đang khởi tạo model...")
    model = recreate_model()

    print(f"Đang load trọng số từ {model_path}...")
    # Load only weights instead of loading the entire model
    model.load_weights(model_path)

    print("Load model thành công!")
except Exception as e:
    print(f"Lỗi khi load model: {e}")
    import traceback
    traceback.print_exc()

# Gán nhãn
labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def preprocess_image(img):
    # Convert to grayscale
    img = img.convert('L')

    # Smart Crop
    # Temporarily invert colors to find the black area of ​​the stroke
    inverted_temp = ImageOps.invert(img)
    bbox = inverted_temp.getbbox()
    if bbox:
        img = img.crop(bbox)

    # RESIZE & PADDING
    target_size = 224
    inner_size = 180  # Inner text size (leave margin on each side ~22px)

    img.thumbnail((inner_size, inner_size), Image.Resampling.LANCZOS)

    # Create a WHITE background
    new_img = Image.new("L", (target_size, target_size), 255)

    # Paste the text in the middle
    offset_x = (target_size - img.size[0]) // 2
    offset_y = (target_size - img.size[1]) // 2
    new_img.paste(img, (offset_x, offset_y))

    # Soften the image
    new_img = new_img.filter(ImageFilter.GaussianBlur(radius=2))

    # Convert to RGB & Normalize
    img_final = new_img.convert('RGB')
    x = np.array(img_final)

    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)

    return x


# def preprocess_image(img):
#     # Chuyển đổi ảnh sang RGB để tránh lỗi kênh Alpha (PNG)
#     img = img.convert('RGB').resize((224, 224))
#     img_array = np.array(img)
#     img_array = img_array.astype('float32') / 255.0
#     img_array = img_array.reshape(1, 224, 224, 3)
#     return img_array


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files and 'drawing' not in request.form:
        return jsonify({'error': 'No image provided'}), 400

    img = None
    img_path = None

    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                img = Image.open(filepath)
                img_path = filepath

        elif 'drawing' in request.form:
            img_data = request.form['drawing'].split(',')[1]
            img = Image.open(io.BytesIO(base64.b64decode(img_data)))
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background

        if img:
            img_array = preprocess_image(img)

            pred = model.predict(img_array)
            pred_class = np.argmax(pred)
            prediction = labels[pred_class]
            confidence = float(np.max(pred))

            return jsonify({
                'prediction': prediction,
                'confidence': confidence,
                'img_path': img_path.replace('\\', '/') if img_path else None
            })

    except Exception as e:
        print(f"Lỗi khi dự đoán: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Could not process image'}), 400


if __name__ == '__main__':
    app.run(debug=True)