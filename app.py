import time
import unicodedata
from string import printable

import pandas as pd
import streamlit as st
import yaml
from greek_accentuation.syllabify import syllabify

from src.cgpos.util.greek import is_greek, is_punctuation
from src.cgpos.util.path import import_pkl

title = "Ancient Greek Part of Speech Tagger"

# Configure page
st.set_page_config(
    page_title=title,
    page_icon="🏺",
)


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
st.title(title)

with st.columns([0.2, 1, 0.2])[1]:
    st.image(image="hydria.jpg", caption="Pictured: Someone trying to learn Greek")

st.subheader("Description", divider=True)

"""
One of the hardest things about learning Ancient Greek is having to memorize [hundreds of word endings](https://en.wiktionary.org/wiki/Appendix:Ancient_Greek_grammar_tables) so you can tell you if a word is a noun, verb, adjective, and so on.

To help make this less painful, I trained a Machine Learning model using the [Ancient Greek and Latin Dependency Treebank](http://perseusdl.github.io/treebank_data/) to predict part of speech for any given word in Ancient Greek.

Select a word from the list below — or enter your own! — to try it out.
"""

st.subheader("Get predictions", divider=True)

text_input = st.toggle("Toggle to enter your own word!")

if text_input:
    # Get word
    input_phrase = "Enter any Ancient Greek word."
    input = st.text_input(
        label=input_phrase,
        placeholder="e.g. λόγος",
    )

else:
    word_list = [
        "ἄνθρωπος",
        "κατηγορῆται",
        "λεγομένων",
        "συμπλοκὴν",
    ]

    input = st.selectbox("Select a word from the list.", word_list)

start = st.button("Go")

result = st.container()

st.subheader("About the model", divider=True)

"""
The underlying model is a [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) trained on syllable-based tokenization. In contrast to many NLP approaches, accents were not removed in order to preserve as much data as possible.

I chose this method because syllables and accent patterns so informative in Greek that I wanted to see how a simple model would perform given the right features. (As it turns out, pretty well!)

Please see the [GitHub repository](https://github.com/tejomaygadgil/cgpos) for details on model performance and implementation.
"""

st.subheader("About the author", divider=True)

st.image("https://tejomaygadgil.github.io/profile.jpg", width=200)

"""
Hi there, I'm [Tejomay](https://tejomaygadgil.github.io/about.html)!

I am passionate about building NLP tools to make it easier to learn language.

Find me on [GitHub](https://github.com/tejomaygadgil), [LinkedIn](https://www.linkedin.com/in/tejomay-gadgil/), or [my blog](https://tejomaygadgil.github.io/)!
"""

# Generate prediction
if start and len(input) > 0:
    if set(input) - set(printable) == set():
        st.write("Please enter a Greek word!")
    else:
        with result, st.spinner("Generating prediction."):
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

            # Format output
            df = pd.DataFrame(pred)
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df[1:]

            time.sleep(1.25)

        result.metric("", df.iloc[0, 0])
        result.dataframe(df.iloc[:, 1:], hide_index=True)
