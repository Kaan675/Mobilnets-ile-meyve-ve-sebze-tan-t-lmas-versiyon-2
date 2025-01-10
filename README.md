# Mobilnets-ile-meyve-ve-sebze-tan-t-lmas-versiyon-2


bu sistem ile kameraya gösterilen meyve ve sebzenin adını söylemektedir ( demo modelidir )  
Bu sistem ile kameraya gösterilen meyve ve sebzenin adını söylemektedir ( demo modelidir ) 
markdown
# Yiyecek Tanıma Modeli
Bu proje, MobileNetV2'yi kullanarak yiyecekleri sınıflandıran bir Python uygulamasıdır. Uygulama, kameradan aldığı görüntüleri işleyerek tanıdığı yiyeceğin ismini döndürür.
## Gereksinimler
Bu projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- `tensorflow`
- `opencv-python`
- `numpy`
## Kurulum
Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:
```bash
pip install tensorflow opencv-python numpy
Kullanım
Uygulamayı çalıştırmak için yiyecek_tanima.py dosyasını aşağıdaki komutla çalıştırabilirsiniz:
bash
python yiyecek_tanima.py
Kod Açıklamaları
Bu bölümde, kodun nasıl çalıştığını anlatan kısa açıklamalar yer almaktadır:
Model Yükleme
MobileNetV2 modelini ImageNet ağırlıkları ile yüklüyoruz:
markdown
# Örnek Veri Seti Klasör Yapısı Oluşturma
Bu proje, bir veri seti için eğitim ve doğrulama veri klasörlerini otomatik olarak oluşturmanızı sağlar. Python kullanarak veri seti klasör yapısını oluşturur ve görüntüleri doğru klasörlere kopyalamanızı kolaylaştırır.
## Gereksinimler
Bu projeyi çalıştırmak için herhangi bir ek kütüphane yüklemenize gerek yoktur, ancak aşağıdaki kütüphaneler Python'un yerleşik modülleri olarak gerekmektedir:
- `os`
- `shutil`
## Kullanım
Uygulamayı çalıştırmak için `dataset_structure.py` dosyasını aşağıdaki komutla çalıştırabilirsiniz:
```bash
python dataset_structure.py
Kod Açıklamaları
Klasör Yapısının Oluşturulması
Eğitim ve doğrulama veri setleri için gerekli klasörleri oluştururuz:
python
dataset_dir = 'dataset'
os.makedirs(os.path.join(dataset_dir, 'train/elma'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/muz'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/brokoli'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/portakal'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/salatalık'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/domates'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/havuç'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'train/kereviz'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/elma'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/muz'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/brokoli'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/portakal'), exist.ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/salatalık'), exist.ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/domates'), exist.ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/havuç'), exist.ok=True)
os.makedirs(os.path.join(dataset_dir, 'validation/kereviz'), exist.ok=True)
Örnek Görüntülerin Kopyalanması
Örnek görüntülerinizi ilgili klasörlere manuel olarak kopyalamanız gerekmektedir. Bu adımı gerçekleştirmek için elma görüntülerinizi train/elma klasörüne, muz görüntülerinizi train/muz klasörüne, vb. yerleştirin.
markdown
# MobileNet ile Yiyecek Sınıflandırma Modeli
Bu proje, MobileNet kullanarak bir görüntü sınıflandırma modeli oluşturmanızı sağlar. Model, belirli yiyecek kategorilerini tanıyabilir ve sınıflandırabilir.
## Gereksinimler
Bu projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- `tensorflow`
- `numpy`
- `opencv-python`
## Kurulum
Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:
```bash
pip install tensorflow numpy opencv-python
Veri Hazırlığı
Bu proje için bir veri setine ihtiyacınız var. Veri setinizi uygun klasör yapısına göre ayarlayın:
dataset/
    train/
        elma/
        muz/
        brokoli/
        portakal/
        salatalık/
        domates/
        havuç/
        kereviz/
    validation/
        elma/
        muz/
        brokoli/
        portakal/
        salatalık/
        domates/
        havuç/
        kereviz/
Her klasöre, ilgili yiyeceklerin görüntülerini yerleştirin.
Kullanım
Uygulamayı çalıştırmak için food_classification.py dosyasını aşağıdaki komutla çalıştırabilirsiniz:
bash
python food_classification.py
Kod Açıklamaları
Veri Artırma
Görüntülerin veri artırma teknikleri ile işlenmesi:
python
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
Veri Yükleme
Eğitim ve doğrulama veri setlerinin yüklenmesi:
python
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\EXCALİBUR\Desktop\Dosyalar\mobilnets\dataset\train', # kendi bilgisayarınızda data set dosyanız nerede ise o yolu kodda değiştirmeniz gerekmektedir
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    r'C:\Users\EXCALİBUR\Desktop\Dosyalar\mobilnets\dataset\validation', # kendi bilgisayarınızda data set dosyanız nerede ise o yolu kodda değiştirmeniz gerekmektedir
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical')
Modelin Yapılandırılması
MobileNet modelinin yapılandırılması ve özelleştirilmesi:
python
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
num_classes = 8
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Modelin Eğitilmesi
Modeli eğitmek ve doğrulamak:
python
model.fit(train_generator, epochs=10, validation_data=validation_generator)
loss, accuracy = model.evaluate(validation_generator)
print(f'Doğruluk: {accuracy}, Kayıp: {loss}')
Modelin Yeniden Eğitilmesi
Modelin belirli katmanlarını yeniden eğitmek:
python
for layer in base_model.layers[:100]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=validation_generator)
Modelin Kaydedilmesi
Eğitim sonrası modelin kaydedilmesi:
python
python
# Örnek görüntüleri doğru klasörlere kopyalama
python
model = tf.keras.applications.MobileNetV2(weights='imagenet')
Sınıf Etiketleri
Yiyecek sınıflarını tanımlıyoruz:
python
class_labels = {948: 'Elma', 966: 'Muz', 906: 'Brokoli', 954: 'Portakal', 943: 'Salatalık', 932: 'Domates', 936: 'Havuç', 937: 'Kereviz'}
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
Görüntü İşleme
Kameradan alınan görüntüleri modelin kabul edebileceği formata getiriyoruz:
python
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image
Görüntü Sınıflandırma
İşlenmiş görüntüyü sınıflandırıyoruz ve en olası yiyecek sınıfını döndürüyoruz:
python
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
Kameradan Görüntü Alma
Kameradan görüntü yakalayıp sınıflandırma işlemi yapıyoruz:
python
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
