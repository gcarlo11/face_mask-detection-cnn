import numpy as np
from flask import Flask, render_template, request, url_for
from PIL import Image
import tensorflow as tf
import io
import os
import uuid  # Untuk membuat nama file unik

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Model TensorFlow yang telah dilatih
model = tf.keras.models.load_model('my_model.keras')  # Pastikan model Anda sudah ada

# Daftar kelas
class_names = ["Mask Weared Incorrectly", "With Mask", "Without Mask"]

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Halaman untuk menerima gambar dan memberikan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        # Membaca gambar dari file yang diunggah
        img = Image.open(file.stream)

        # Menyimpan gambar dengan nama unik
        image_filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        img.save(image_path)

        # Proses gambar seperti yang dilakukan sebelumnya
        image = img.resize((35, 35))
        image_array = np.array(image) / 255.0  # Normalisasi
        image_array = np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension

        # Melakukan prediksi
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions)]  # Indeks kelas yang diprediksi
        confidence = np.max(predictions)  # Confidence level

        return render_template('index.html', predicted_class=predicted_class, confidence=confidence, image_filename=image_filename)

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
