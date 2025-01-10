

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore # type: ignore
from tensorflow.keras.applications import MobileNet # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore

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

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\EXCALİBUR\Desktop\Dosyalar\mobilnets\dataset\train', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    r'C:\Users\EXCALİBUR\Desktop\Dosyalar\mobilnets\dataset\validation', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical')

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
num_classes = 8
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # num_classes, sınıf sayısıdır

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=20, validation_data=validation_generator)

loss, accuracy = model.evaluate(validation_generator)
print(f'Doğruluk: {accuracy}, Kayıp: {loss}')

for layer in base_model.layers[:100]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=20, validation_data=validation_generator)

model.save(r'C:\Users\EXCALİBUR\Desktop\Dosyalar\mobilnets\savemodel3.h5')
