import os
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class IMDbDataLoader:
    def __init__(self, dataset_path, max_vocab=10000, max_len=200):
        self.dataset_path = dataset_path
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")

    def load_data(self, limit_per_class=1000):
        texts = []
        labels = []

        for label_type in ['pos', 'neg']:
            dir_name = os.path.join(self.dataset_path, 'train', label_type)
            file_list = os.listdir(dir_name)
            file_list = file_list[:limit_per_class]  # Limit samples per class

            for fname in file_list:
                if fname.endswith('.txt'):
                    with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                        texts.append(f.read())
                    labels.append(1 if label_type == 'pos' else 0)

        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts[:], labels[:] = zip(*combined)
        return list(texts), list(labels)

    def tokenize_and_pad(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded

    def prepare_data(self):
        texts, labels = self.load_data(limit_per_class=1000)  # Load 1000 pos + 1000 neg = 2000
        padded = self.tokenize_and_pad(texts)
        X_train, X_test, y_train, y_test = train_test_split(
            padded, labels, test_size=0.2, random_state=42)

        return (np.array(X_train), np.array(X_test),
                np.array(y_train), np.array(y_test),
                self.tokenizer)