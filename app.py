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
features_map, labels_long = load_maps(config)
model = load_model(config)

# Format labels
labels_long = [
    [label.capitalize() if label != "N/A" else label for label in category]
    for category in labels_long
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
"""
# Ancient Greek Part of Speech Tagger

One of the hardest things about learning Ancient Greek is having to memorize word ending tables so you know what is a verb, noun, and so on.

This app is trained on ___. Enter in Ancient Greek word to find its part of speech!
"""

# Get word
input_phrase = "Enter a word here"
input = st.text_input(
    label=input_phrase, value=input_phrase, label_visibility="collapsed"
)

if len(input) > 0:
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
        tokens = [features_map[syllable] for syllable in syllables]

        # Get prediction
        pred = model.predict([tokens])
        pred = [
            [classes[i], labels_long[i][value]] for i, value in enumerate(pred[0])
        ]  # Get text label
        pred = [pred[i] for i in reorder_map]  # Reorder
        pred = [value for value in pred if value[1] != "N/A"]

        df = pd.DataFrame(pred)
        df = df.transpose()
        df.columns = df.iloc[0]
        df = df[1:]

        st.dataframe(df, hide_index=True)
