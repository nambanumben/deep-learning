import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle
import random

nltk.download('brown')
from nltk.corpus import brown

sentences = [" ".join(sent) for sent in brown.sents()][:20000]  # up to 20k sentences

# No OOV token here
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)
total_words = 5000

input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, min(len(token_list), 10)):  # max seq length 10
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_seq_len - 1))
model.add(LSTM(128))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=10, verbose=1)

model.save("better_model.h5")
with open("better_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("better_info.txt", "w") as f:
    f.write(f"max_seq_len={max_seq_len}\n")
