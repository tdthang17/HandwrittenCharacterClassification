import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A_caps", "B_caps", "C_caps", "D_caps", "E_caps", "F_caps", "G_caps", "H_caps", "I_caps", "J_caps",
    "K_caps", "L_caps", "M_caps", "N_caps", "O_caps", "P_caps", "Q_caps", "R_caps", "S_caps", "T_caps",
    "U_caps", "V_caps", "W_caps", "X_caps", "Y_caps", "Z_caps",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z"
]


model = tf.keras.models.load_model("model.keras")
print("Ok")

img_path = "img.png"
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
predicted_index = np.argmax(pred[0])
confidence = np.max(pred[0])

plt.imshow(img)
plt.axis('off')
plt.title(f"Dự đoán: {CLASS_NAMES[predicted_index]}\nĐộ tin cậy: {confidence:.2%}")
plt.show()
