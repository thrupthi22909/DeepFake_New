import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# --------------------------
# Load preprocessed dataset
# --------------------------
data = np.load("dataset2_processed.npz", allow_pickle=True)
X, y = data["X"], data["y"]

# Shuffle dataset
idx = np.arange(len(X))
np.random.shuffle(idx)
X, y = X[idx], y[idx]

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("✅ Training data:", X_train.shape, "Test:", X_test.shape)

# --------------------------
# Build improved CNN+LSTM model
# --------------------------
model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'), input_shape=(10, 128, 128, 3)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2,2))),

    TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2,2))),

    TimeDistributed(Conv2D(128, (3,3), activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2,2))),

    TimeDistributed(Flatten()),
    LSTM(128, dropout=0.3, recurrent_dropout=0.3),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# --------------------------
# Compile model
# --------------------------
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------
# Callbacks for stability
# --------------------------
checkpoint = ModelCheckpoint("video_best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)

# --------------------------
# Train model
# --------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=4,
    callbacks=[checkpoint, lr_reduce, early_stop],
    verbose=1
)

# --------------------------
# Evaluate
# --------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"🎯 Test Accuracy: {acc*100:.2f}%")
print("✅ Training complete! Best model saved as video_best_model.h5")
