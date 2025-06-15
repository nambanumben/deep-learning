import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# Toy dataset
toy_sentences = [
    "the cat sits on the mat",
    "the dog sits on the mat",
    "the cat jumps on the mat",
    "the dog jumps on the mat",
    "the cat runs on the mat",
    "the dog runs on the mat"
] * 100  # Repeat to give more data

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(toy_sentences)
total_words = len(tokenizer.word_index) + 1

# Build input sequences
input_sequences = []
for line in toy_sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Model
model = Sequential()
model.add(Embedding(total_words, 16, input_length=max_seq_len - 1))
model.add(LSTM(32))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train (it will overfit, that's what we want for a demo)
model.fit(X, y, epochs=40, verbose=2)  # More epochs = higher accuracy

model.save("toy_model.h5")
with open("toy_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("toy_info.txt", "w") as f:
    f.write(f"max_seq_len={max_seq_len}\n")

# Evaluate on train set (should be very high)
loss, acc = model.evaluate(X, y, verbose=0)
print(f"\nTrain accuracy: {acc*100:.2f}%")
