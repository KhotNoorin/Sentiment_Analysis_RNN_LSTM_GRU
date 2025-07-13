import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils.data_loader import IMDbDataLoader
from models.bigru import build_bigru_model

# Hyperparameters
MAX_VOCAB = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5
MODEL_SAVE_PATH = 'checkpoints/bigru_model.h5'
TOKENIZER_SAVE_PATH = 'checkpoints/bigru_tokenizer.pkl'

# 1. Load IMDb data
print("🔄 Loading IMDb dataset...")
loader = IMDbDataLoader(dataset_path='data/raw/aclImdb', max_vocab=MAX_VOCAB, max_len=MAX_LEN)
X_train, X_test, y_train, y_test, tokenizer = loader.prepare_data()

# 2. Build BiGRU model
print("🔧 Building BiGRU model...")
model = build_bigru_model(vocab_size=MAX_VOCAB, embedding_dim=EMBEDDING_DIM, input_length=MAX_LEN)

# 3. Train BiGRU
print("🚀 Training BiGRU model...")
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS)

# 4. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# 5. Save model and tokenizer
os.makedirs('checkpoints', exist_ok=True)
model.save(MODEL_SAVE_PATH)

with open(TOKENIZER_SAVE_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)

# 6. Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('checkpoints/bigru_training_plot.png')
    plt.show()

plot_history(history)
