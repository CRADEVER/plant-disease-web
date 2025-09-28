# 1️⃣ Mount Google Drive (tùy chọn)
from google.colab import drive
drive.mount('/content/drive')

# 2️⃣ Cài đặt thư viện
!pip install tensorflow matplotlib tensorflowjs --quiet

# 3️⃣ Import thư viện
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# 4️⃣ Đường dẫn dataset & model
DATASET_PATH = '/content/drive/MyDrive/dataset'  # Thay bằng đường dẫn của bạn
MODEL_PATH = '/content/drive/MyDrive/plant_model.keras' # Changed to .keras format
TFJS_PATH = '/content/drive/MyDrive/plant_model_js'

# 5️⃣ Tham số
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

# 6️⃣ Data generator
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("Number of classes:", num_classes)
print("CLASS_NAMES =", list(train_gen.class_indices.keys()))

# 7️⃣ Tạo mới model (luôn tạo mới để tránh lỗi output)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 8️⃣ Train model
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# 9️⃣ Vẽ đồ thị
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')
plt.show()

# 🔟 Convert sang TensorFlow.js
!tensorflowjs_converter --input_format keras {MODEL_PATH} {TFJS_PATH}
print("TensorFlow.js model saved to:", TFJS_PATH)

# 1️⃣1️⃣ In ra CLASS_NAMES cho web
print("CLASS_NAMES =", list(train_gen.class_indices.keys()))