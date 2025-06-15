import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import gradio as gr

# Load model, tokenizer, and max_seq_len
model = tf.keras.models.load_model("better_model.h5")
with open("better_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("better_info.txt", "r") as f:
    max_seq_len = int(f.read().split("=")[1])

def predict_top_k_next_words_better(seed_text, k=3, temperature=1.0):
    seed_text = seed_text.strip().lower()
    if not seed_text:
        return ""
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    probs = model.predict(token_list, verbose=0)[0]

    # Temperature scaling for diversity
    if temperature != 1.0:
        probs = np.log(probs + 1e-8) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)
    
    # Get top k indices
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
    fn=predict_top_k_next_words_better,
    inputs=gr.Textbox(lines=1, placeholder="Type a sentence..."),
    outputs="text",
    live=True,
    title="🔮 Better Next Word Prediction (LSTM)",
    description="Type a sentence and get next-word predictions from a model trained on the Brown corpus!"
)

iface.launch()
