"""Elektrik izolatörlerinde CNN tabanlı hasar sınıflandırma örnek kodu.

Not: Çalıştırmadan önce README veya rapordaki klasör yapısına göre dataset klasörlerini hazırlayın.
Bu kodun sonuçları, gerçek veri seti üzerinde çalıştırılmadan doğrulanmış sayılmaz.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Temel ayarlar
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# 2. Eğitim, doğrulama ve test verilerini yükleme
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["healthy", "damaged"],
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["healthy", "damaged"],
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["healthy", "damaged"],
    shuffle=False
)

# 3. Veri yükleme performansını iyileştirme
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 4. CNN modelini oluşturma
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# 5. Modeli derleme
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Erken durdurma
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# 7. Model eğitimi
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping]
)

# 8. Test verisi üzerinde değerlendirme
test_loss, test_accuracy = model.evaluate(test_ds)

print("Test kaybı:", test_loss)
print("Test doğruluğu:", test_accuracy)

# 9. Test tahminlerini oluşturma
y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

y_prob = model.predict(test_ds, verbose=0).reshape(-1)
y_pred = (y_prob >= 0.5).astype(int)

# 10. Sınıflandırma raporu
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Hasarsız", "Hasarlı"],
        zero_division=0
    )
)

# 11. Karmaşıklık matrisi
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Hasarsız", "Hasarlı"]
)

display.plot(values_format="d")
plt.title("İzolatör Hasar Tespiti - Karmaşıklık Matrisi")
plt.tight_layout()
plt.show()

# 12. Eğitim ve doğrulama doğruluğu
plt.figure()
plt.plot(history.history["accuracy"], label="Eğitim")
plt.plot(history.history["val_accuracy"], label="Doğrulama")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.title("Eğitim ve Doğrulama Doğruluğu")
plt.legend()
plt.tight_layout()
plt.show()

# 13. Eğitim ve doğrulama kaybı
plt.figure()
plt.plot(history.history["loss"], label="Eğitim")
plt.plot(history.history["val_loss"], label="Doğrulama")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.title("Eğitim ve Doğrulama Kaybı")
plt.legend()
plt.tight_layout()
plt.show()

# 14. Eğitilen modeli kaydetme
model.save("izolator_hasar_modeli.keras")
