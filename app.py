import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(
    page_title="Klasifikasi Ras Anjing",
    layout="centered",
    page_icon="🐾"
)

# Direktori Galeri
GALERI_DIR = "galeri"
os.makedirs(GALERI_DIR, exist_ok=True)

# CSS Kustom
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #ffffff;
}
.title {
    text-align: center;
    font-size: 36px;
    color: #4F8BF9;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: #999999;
    margin-top: 30px;
}
.stButton > button {
    background-color: #4F8BF9;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
.stFileUploader {
    border: 2px dashed #4F8BF9;
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 20px;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🐶 Aplikasi Klasifikasi Ras Anjing</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unggah gambar anjing dan biarkan AI memprediksi rasnya!</div>', unsafe_allow_html=True)
st.markdown("---")

# Informasi Umum Aplikasi
st.markdown("## ℹ️ Tentang Aplikasi")
st.markdown("""
Aplikasi **Klasifikasi Ras Anjing** ini dikembangkan menggunakan teknologi *Artificial Intelligence*, khususnya *Deep Learning* dengan arsitektur **MobileNetV2**.

Aplikasi ini mampu mengenali dan mengklasifikasikan gambar anjing ke dalam 5 ras berbeda:
- 🐾 French Bulldog
- 🐾 German Shepherd
- 🐾 Golden Retriever
- 🐾 Poodle
- 🐾 Yorkshire Terrier

Dengan bantuan model cerdas berbasis **Transfer Learning**, sistem ini dapat mengidentifikasi jenis ras anjing dengan tingkat kepercayaan (*confidence*) tinggi, hanya dari gambar yang diunggah.

### 🔍 Cara Menggunakan:
1. Siapkan gambar anjing berformat .jpg, .jpeg, atau .png.
2. Klik tombol unggah untuk memilih gambar.
3. Aplikasi akan memproses dan menampilkan hasil prediksi lengkap dengan tingkat kepercayaannya.
4. Informasi tentang ras yang dikenali juga akan ditampilkan.

> Aplikasi ini cocok digunakan oleh pecinta hewan peliharaan, pemilik anjing, atau siapa pun yang ingin mengenal lebih dalam tentang karakteristik ras anjing melalui gambar.
""")

# Load Model & Label
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("dog_breed_classifier_final.keras")
    labels = [
        "french_bulldog",
        "german_shepherd",
        "golden_retriever",
        "poodle",
        "yorkshire_terrier"
    ]
    return model, labels

model, class_names = load_model_and_labels()
CONFIDENCE_THRESHOLD = 0.70

# Informasi Setiap Ras
DOG_BREED_DETAILS = {
    "french_bulldog": {
        "nama": "French Bulldog",
        "asal": "Prancis",
        "ukuran": "Kecil (hingga 12 kg)",
        "ciri_khas": "Telinga kelelawar, tubuh kompak, moncong pendek",
        "perilaku": "Ramah, tenang, dan cocok sebagai anjing rumahan.",
        "perawatan": "Membutuhkan perawatan wajah rutin untuk mencegah iritasi pada lipatan kulit."
    },
    "german_shepherd": {
        "nama": "German Shepherd",
        "asal": "Jerman",
        "ukuran": "Besar (30–40 kg)",
        "ciri_khas": "Telinga tegak, bulu tebal, anjing penjaga yang sangat cerdas",
        "perilaku": "Cerdas, setia, dan mudah dilatih, sangat cocok sebagai anjing penjaga.",
        "perawatan": "Perlu disisir secara rutin karena bulunya yang tebal dan mudah rontok."
    },
    "golden_retriever": {
        "nama": "Golden Retriever",
        "asal": "Skotlandia",
        "ukuran": "Sedang (25–34 kg)",
        "ciri_khas": "Bulu emas panjang, ramah, suka air",
        "perilaku": "Sangat bersahabat, penyayang, dan cocok untuk keluarga dengan anak-anak.",
        "perawatan": "Memerlukan aktivitas fisik harian dan perawatan bulu yang teratur."
    },
    "poodle": {
        "nama": "Poodle",
        "asal": "Jerman/Prancis",
        "ukuran": "Mini hingga sedang",
        "ciri_khas": "Bulu keriting, sangat cerdas dan mudah dilatih",
        "perilaku": "Aktif, cerdas, dan mudah beradaptasi dengan lingkungan baru.",
        "perawatan": "Bulu keriting perlu dipangkas dan dirawat secara rutin agar tidak kusut."
    },
    "yorkshire_terrier": {
        "nama": "Yorkshire Terrier",
        "asal": "Inggris",
        "ukuran": "Kecil (2–3 kg)",
        "ciri_khas": "Bulu panjang dan halus, aktif, cocok di apartemen",
        "perilaku": "Lincah, berani, dan senang menjadi pusat perhatian.",
        "perawatan": "Membutuhkan penyisiran bulu harian dan perhatian khusus pada kesehatan gigi."
    }
}

# Preprocessing Gambar
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Histori Klasifikasi
if "history" not in st.session_state:
    st.session_state.history = []

# Upload & Prediksi
st.markdown("---")
uploaded_file = st.file_uploader("📷 Unggah gambar anjing Anda", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📌 Gambar Anda", use_container_width=True)

    st.markdown("### 🔎 Hasil Prediksi")
    with st.spinner("🔄 Sedang memproses gambar..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0]
        predicted_idx = np.argmax(prediction)
        predicted_class_raw = class_names[predicted_idx]
        predicted_class = predicted_class_raw.lower().replace(" ", "_")
        confidence = prediction[predicted_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        st.success(f"🐾 Prediksi: **{predicted_class_raw.replace('_', ' ').title()}**")
        st.metric("🎯 Tingkat Kepercayaan", f"{confidence * 100:.2f}%")

        detail = DOG_BREED_DETAILS.get(predicted_class, {})
        st.markdown(f"""
        <div style='
            background-color: #1e1e1e;
            padding: 20px;
            border-left: 5px solid #4F8BF9;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 2px 2px 5px rgba(255,255,255,0.05);
            color: #ffffff;
        '>
            <h4 style='color:#4F8BF9;'>📘 Informasi Ras: {detail.get("nama", "-")}</h4>
            <p><strong>Asal:</strong> {detail.get("asal", "-")}</p>
            <p><strong>Ukuran:</strong> {detail.get("ukuran", "-")}</p>
            <p><strong>Ciri Khas:</strong> {detail.get("ciri_khas", "-")}</p>
            <p><strong>Perilaku:</strong> {detail.get("perilaku", "-")}</p>
            <p><strong>Perawatan:</strong> {detail.get("perawatan", "-")}</p>

        </div>
        """, unsafe_allow_html=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        img_path = os.path.join(GALERI_DIR, f"anjing_{timestamp}.jpg")
        image.save(img_path)

        st.session_state.history.append({
            "timestamp": timestamp,
            "label": predicted_class_raw.replace("_", " ").title(),
            "confidence": f"{confidence * 100:.2f}%",
            "img_path": img_path
        })
    else:
        st.error("⚠️ Gambar tidak dikenali")
        st.write("Model tidak yakin bahwa gambar ini termasuk dalam 5 ras anjing yang dikenali.")

# Galeri Gambar
st.markdown("### 🖼️ Galeri Gambar")
img_files = sorted(os.listdir(GALERI_DIR), reverse=True)[:5]
if img_files:
    cols = st.columns(len(img_files))
    for col, file in zip(cols, img_files):
        col.image(os.path.join(GALERI_DIR, file), use_container_width=True)
else:
    st.info("Belum ada gambar dalam galeri.")

# Riwayat Klasifikasi
st.markdown("### 📜 Riwayat Klasifikasi")
if st.session_state.history:
    for item in reversed(st.session_state.history[-5:]):
        st.markdown(f"""
        <div style='
            padding: 10px;
            background-color: #1e1e1e;
            border-radius: 8px;
            margin-bottom: 10px;
            color: #ffffff;
        '>
            🧒 <strong>{item['timestamp']}</strong><br/>
            📷 <em>{os.path.basename(item['img_path'])}</em><br/>
            🐕 <strong style='color:#4F8BF9;'>{item['label']}</strong><br/>
            🎯 Keyakinan: <strong>{item['confidence']}</strong>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Belum ada riwayat klasifikasi.")

# Footer
st.markdown('<div class="footer">© 2025 Aplikasi Klasifikasi Anjing | Dibuat oleh Mita Mukaromah</div>', unsafe_allow_html=True)