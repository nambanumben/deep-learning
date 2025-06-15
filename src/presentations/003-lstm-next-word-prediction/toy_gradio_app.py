import gradio as gr
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("toy_model.h5")
with open("toy_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("toy_info.txt", "r") as f:
    max_seq_len = int(f.read().split("=")[1])

def predict_next_word(seed_text, k=3):
    seed_text = seed_text.strip().lower()
    if not seed_text:
        return ""
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    probs = model.predict(token_list, verbose=0)[0]
    top_indices = probs.argsort()[-k:][::-1]
    inv_index = {v: k for k, v in tokenizer.word_index.items()}
    suggestions = []
    for idx in top_indices:
        word = inv_index.get(idx)
        if word and word not in suggestions:
            suggestions.append(word)
    if suggestions:
        return "\n".join([f"{seed_text} {word}" for word in suggestions])
    return "No predictions found."

iface = gr.Interface(
    fn=predict_next_word,
    inputs=gr.Textbox(lines=1, placeholder="Type a sentence..."),
    outputs="text",
    live=True,
    title="🐱🐶 Toy LSTM Next-Word Predictor",
    description="Type a sentence fragment (e.g., 'the cat') and see next word predictions from a toy LSTM trained on simple sentences."
)

iface.launch()

