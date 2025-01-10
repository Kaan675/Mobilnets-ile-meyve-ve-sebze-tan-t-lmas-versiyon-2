
    
    
import cv2 # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
import numpy as np # type: ignore

# MobileNetV2 modelini yükle
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Model için sınıf etiketleri
class_labels = {948: 'Elma', 966: 'Muz', 906: 'Brokoli', 954: 'Portakal', 943: 'Salatalık', 932: 'Domates', 936: 'Havuç', 937: 'Kereviz'}

# Besin değerleri
nutrition_info = {
    'Elma': {'Kalori': 52, 'Yağ': 0.2, 'Protein': 0.3},
    'Muz': {'Kalori': 89, 'Yağ': 0.3, 'Protein': 1.1},
    'Brokoli': {'Kalori': 34, 'Yağ': 0.4, 'Protein': 2.8},
    'Portakal': {'Kalori': 47, 'Yağ': 0.1, 'Protein': 0.9},
    'Salatalık': {'Kalori': 16, 'Yağ': 0.1, 'Protein': 0.7},
    'Domates': {'Kalori': 18, 'Yağ': 0.2, 'Protein': 0.9},
    'Havuç': {'Kalori': 41, 'Yağ': 0.2, 'Protein': 0.9},
    'Kereviz': {'Kalori': 16, 'Yağ': 0.2, 'Protein': 0.7}
}

# Bir görüntüyü işle
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Görüntüyü sınıflandır
def classify_image(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    top_prediction = np.argmax(predictions)
    food_item = class_labels.get(top_prediction, 'yiyecek taninamadi')
    nutrition = nutrition_info.get(food_item, {'Kalori': 'Bilinmiyor', 'Yağ': 'Bilinmiyor', 'Protein': 'Bilinmiyor'})
    return food_item, nutrition

# Meyve veya sebze olup olmadığını belirle
def fruit_or_vegetable(label):
    fruits = ['Elma', 'Muz', 'Portakal']
    vegetables = ['Brokoli', 'Salatalık', 'Domates', 'Havuç', 'Kereviz']
    
    if label in fruits:
        return 'meyve'
    elif label in vegetables:
        return 'sebze'
    else:
        return 'taninamadi'

# Kameradan görüntü al
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açilmadi!")
        return None

    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Görüntü yakalanamadi!")
        return None

# Görüntüyü sınıflandır ve görsel geri bildirim ekle
image = capture_image_from_camera()
if image is not None:
    result, nutrition = classify_image(image)
    category = fruit_or_vegetable(result)
    output_text = f"Bu yiyecek: {result}, bu bir {category}. Kalori: {nutrition['Kalori']} kcal, Yağ: {nutrition['Yağ']} g, Protein: {nutrition['Protein']} g."
    
    # Görüntü üzerine metin ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, output_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Siniflandirilan Görüntü", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Görüntü alinamadi!")

