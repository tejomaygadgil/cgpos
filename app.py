import unicodedata

import streamlit as st
import yaml
from greek_accentuation.syllabify import syllabify

from src.cgpos.util.greek import is_greek, is_punctuation
from src.cgpos.util.path import import_pkl

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load files
features_map = import_pkl(config["reference"]["features_map"])
targets_map = import_pkl(config["reference"]["targets_map"])
model = import_pkl(config["model"])
# Get word
input = st.text_input("Enter an Ancient Greek word to find its part of speech.")

if len(input) > 0:
    # TODO check word is greek

    # Normalize
    form = unicodedata.normalize("NFD", input)
    form = "".join([char for char in form if (is_greek(char) or is_punctuation(char))])
    form = unicodedata.normalize("NFC", form)

    # TODO Make sure it's only one word

    # Syllablize
    syllables = syllabify(form)

    # Tokenize
    tokens = [features_map[syllable] for syllable in syllables]

    # Get prediction
    pred = model.predict([tokens])
    pp_pred = [targets_map[2][i][value] for i, value in enumerate(pred[0])]
    # pp_pred = targets_map

    st.write(input)
    st.write([form])
    st.write([syllables])
    st.write([tokens])
    st.write(pred)
    st.write(pp_pred)