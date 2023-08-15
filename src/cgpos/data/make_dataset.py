# -*- coding: utf-8 -*-
import logging
import os
import unicodedata
import xml.etree.ElementTree as ET

import hydra
from greek_accentuation.syllabify import syllabify
from omegaconf import DictConfig

from cgpos.utils.util import (
    export_pkl,
    get_abs_dir,
    import_pkl,
    is_greek,
    is_punctuation,
)


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def parse(config: DictConfig):
    """
    Convert raw Perseus treebank XML data into a tabular format.
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing Perseus data:")

    # Set import and export directories
    import_dir = get_abs_dir(config.perseus.raw_dir)
    export_dir = config.perseus.parsed

    # Get files
    files = os.listdir(import_dir)
    logger.info(f"Importing {len(files)} files from {import_dir}")

    # Collect words from files
    sentence_id = 0  # Create sentence-level ID
    data = []
    for file in files:
        file_dir = os.path.join(import_dir, file)
        logger.info(f"Processing {file}")

        # Load XML
        tree = ET.parse(file_dir)
        root = tree.getroot()

        # Get title and author
        work_attrib = {}
        for field in ["author", "title"]:
            element = list(root.iter(field))
            if len(element):  # Some files are missing author info
                work_attrib[field] = element[0].text

        # Get POS tags
        for sentence in root.iter("sentence"):
            sentence_attrib = sentence.attrib.copy()
            sentence_attrib["sentence_id"] = str(sentence_id)
            sentence_id += sentence_id
            for word in sentence.iter("word"):
                word_attrib = word.attrib.copy()
                word_attrib.update(sentence_attrib)
                word_attrib.update(work_attrib)
                data.append(word_attrib)

    # Export
    export_pkl(data, export_dir)

    logger.info(f"Success! Extracted {len(data)} words from {len(files)} files.")


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def target_map(config: DictConfig):
    """
    Build map to parse target column.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building target map:")

    # Set import and export directories
    file_dir = get_abs_dir(config.perseus.tagset)
    export_dir = config.reference.target_map

    # Load XML
    tree = ET.parse(file_dir)
    root = tree.getroot()

    # Build map
    data = ([], [])
    for element in next(root.iter("attributes")):
        pos_class = element.tag
        data[0].append(pos_class)
        data[1].append([[], []])
        data[1][-1][0].append("-")
        data[1][-1][1].append("N/A")
        for _i, value in enumerate(element.find("values"), start=1):
            short = value.find("postag").text
            long = value.find("long").text
            data[1][-1][0].append(short)
            data[1][-1][1].append(long)

    # Accept irregular pos tag

    # Export
    export_pkl(data, export_dir)
    data[1][0].append(("x", "irregular"))

    logger.info(f"Success! Built target map for {len(data[0])} targets: {data[0]}")


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def normalize(config: DictConfig):
    """
    Normalize Perseus data by
    - Decomposing unicode diacritics (cf. https://www.degruyter.com/document/doi/10.1515/9783110599572-009/html)
    - Stripping non-Greek characters.
    - Recomposing diacritics (for better syllablization)
    """
    logger = logging.getLogger(__name__)
    logger.info("Normalizing Perseus data:")

    # Set import and export directories
    import_dir = config.perseus.parsed
    export_dir = config.perseus.normalized

    # Import data
    data = import_pkl(import_dir)

    # Normalize
    for word in data:
        form = word["form"]
        # Split letter from diacritics (["ί"] becomes ["ι"," ́ "])
        form = unicodedata.normalize("NFD", form)
        # Strip non-Greek chars
        form = "".join(
            [char for char in form if (is_greek(char) or is_punctuation(char))]
        )
        # Recompose (["ι"," ́ "] becomes ["ί"])
        form = unicodedata.normalize("NFC", form)

        word["norm"] = form

    # Export
    export_pkl(data, export_dir)

    logger.info(
        "Success! Performed unicode normalization and stripped non-Greek characters."
    )


@hydra.main(config_path="../../../conf", config_name="main", version_base=None)
def clean(config: DictConfig):
    """
    Clean normalized data for training:

    - Drop malformed words
    - Build targets for training.

    Export: [[uid_1], ...], [[form_1], ...], [[target_1], ...]]
    """
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data for training:")

    # Set import directories
    import_dir = config.perseus.normalized
    target_map_dir = config.reference.target_map
    # Set export directories
    export_dir = config.data.cleaned

    # Import data
    data = import_pkl(import_dir)
    _, target_values = import_pkl(target_map_dir)

    # Normalize
    cleaned = [[], [], []]
    malform = []
    for word in data:
        try:
            assert len(word.get("postag", "")) == 9
            sentence_id = word["sentence_id"]
            # Build features
            norm = word["norm"]
            feature = syllabify(norm)
            # Build target
            target = []
            postag = word["postag"]
            for i, short in enumerate(postag):
                match (i, short):
                    case ("5", "d"):  # Treat depondent verbs as medio-passive
                        i, short = "5", "e"
                value = target_values[i][0].index(short)
                target.append(value)
            # Append
            cleaned[0].append(sentence_id)
            cleaned[1].append(feature)
            cleaned[2].append(target)
        except (AssertionError, ValueError):
            malform.append(word)

    n = len(cleaned[0])
    length_match = [len(collection) == n for collection in cleaned]
    assert (
        length_match
    ), f"Cleaned lengths ({len(cleaned[0])}, {len(cleaned[1])}, {len(cleaned[2])}) do not match."

    # Export
    export_pkl(cleaned, export_dir)

    logger.info(
        f"Success! Exporting {len(cleaned[0])} words (dropped {len(malform)} malformed words)."
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    parse()
    target_map()
    normalize()
    clean()
