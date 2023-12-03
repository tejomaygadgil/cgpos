import time
import unicodedata
from string import printable

import pandas as pd
import streamlit as st
import yaml
from greek_accentuation.syllabify import syllabify

from src.cgpos.util.greek import is_greek, is_punctuation
from src.cgpos.util.path import import_pkl


# Define functions with caching to improve performance
@st.cache_data
def get_config():
    """
    Returns project config file.
    """
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


@st.cache_data
def load_maps(config):
    """
    Return tokenization and labels mapping.
    """
    features_map = import_pkl(config["reference"]["features_map"])
    _, _, labels_long = import_pkl(config["reference"]["targets_map"])
    return features_map, labels_long


@st.cache_resource
def load_model(config):
    """
    Return model.
    """
    model = import_pkl(config["model"])
    return model


# Load data
config = get_config()
syl2tok, labels = load_maps(config)
model = load_model(config)

# Format labels
labels = [
    [label.capitalize() if label != "N/A" else label for label in category]
    for category in labels
]
classes = [
    "Part of speech",
    "Person",
    "Number",
    "Tense",
    "Mood",
    "Voice",
    "Gender",
    "Case",
    "Degree",
]
reorder_map = [0, 6, 1, 2, 7, 3, 4, 5, 8]

# Start app
st.title("Ancient Greek Part of Speech Tagger")

st.subheader("Description", divider=True)

"""
One of the hardest things about learning Ancient Greek is having to memorize hundreds of word endings so you can recognize nouns, adjectives, verbs and so on.

This app is trained on ___. Enter in Ancient Greek word to find its part of speech!
"""

# Get word
input_phrase = "Enter a word here"
# input = st.text_input(
#    label=input_phrase,
#    value=input_phrase,
#    label_visibility="collapsed",
# )

word_list = [
    "ἄνθρωπος",
    "κατηγορῆται",
    "λεγομένων",
    "συμπλοκὴν",
]

st.selectbox("Select something", word_list, key="input")
start = st.button("Go")
input = st.session_state.input

# with st.expander("See explanation"):
#    st.write(
#        """
#             The chart above shows some numbers I picked for you.
#             I rolled actual dice for these, so they're *guaranteed* to
#             be random.
#             """
#    )
#    st.image("https://static.streamlit.io/examples/dice.jpg")

if start and len(input) > 0:
    if set(input) - set(printable) == set():
        if input != input_phrase:
            st.write("Please enter a Greek word!")
    else:
        # Normalize
        form = unicodedata.normalize("NFD", input)
        form = "".join(
            [char for char in form if (is_greek(char) or is_punctuation(char))]
        )
        form = unicodedata.normalize("NFC", form)

        # TODO Make sure it's only one word

        # Convert syllables to tokens
        syllables = syllabify(form)
        tokens = [syl2tok[syllable] for syllable in syllables]

        # Get prediction
        pred = model.predict([tokens])
        pred = [
            [classes[i], labels[i][value]] for i, value in enumerate(pred[0])
        ]  # Get text label
        pred = [pred[i] for i in reorder_map]  # Reorder
        pred = [value for value in pred if value[1] != "N/A"]

        df = pd.DataFrame(pred)
        df = df.transpose()
        df.columns = df.iloc[0]
        df = df[1:]

        bar = st.progress(0)
        for i in range(100):
            # Update the progress bar with each iteration.
            time.sleep(1e-2)
            bar.progress(i + 1, text="Running model")

        time.sleep(0.5)
        st.table(df)
        bar.progress(100, text="Done!")
