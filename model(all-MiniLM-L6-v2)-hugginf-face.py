import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# HuggingFace API Ayarları
API_TOKEN = "Hugging_Face_Token"  # HuggingFace'den aldığınız Token
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace'deki model ismi
API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_NAME}"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# 1. Örnek Veritabanı: Reklam Metinleri
advertisements = [
    "Asus ZenBook laptop, hafif ve taşınabilir. Hemen sipariş ver!",
    "MacBook Air ile performans ve şıklığı bir arada yaşayın.",
    "Lenovo ThinkPad, iş dünyasının vazgeçilmez laptopu.",
    "MSI oyun laptoplarıyla yüksek performans ve hız.",
    "Dell XPS serisiyle tasarım ve güçlü donanım bir arada.",
    "Suyu sıcak tutan harkia bir Termos. Kışları çayınızı sıcak tutar.",
    
]

# 2. HuggingFace Embedding Fonksiyonu
def get_embeddings(text_list):
    """Metin listesini HuggingFace API kullanarak embedding vektörlerine çevirir."""
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text_list})
    if response.status_code == 200:
        print(response.json())
        return np.array([result for result in response.json()])
    else:
        raise Exception(f"API Hatası: {response.status_code}, {response.text}")

# 3. Reklamları Vektörleştir (Önceden Embed Edilmiş Veriler)
print("Reklam metinleri vektörleştiriliyor...")
advertisement_embeddings = get_embeddings(advertisements)

# 4. Kullanıcı Girişi
user_input = input("Telegram mesajını girin: ")

# Kullanıcı mesajını vektörleştir
print("Kullanıcı mesajı vektörleştiriliyor...")
user_embedding = get_embeddings([user_input])[0]

# 5. Cosine Similarity Hesaplama
similarities = cosine_similarity([user_embedding], advertisement_embeddings)

# 6. En Uygun Reklamı Bul
top_match_index = np.argmax(similarities)
top_match_score = similarities[0][top_match_index]

# 7. Sonuç Göster
print("\nEn uygun reklam:")
print(f"- {advertisements[top_match_index]} (Benzerlik Skoru: {top_match_score:.2f})")
