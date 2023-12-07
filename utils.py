"""
Utility functions
"""
import tensorflow as tf
import os
from collections import Counter
import numpy as np


def get_surrounding(sentence, window=3):
    """
    Extracts surrounding context for each word in a sentence based on a specified window size.

    Parameters:
    - sentence (list): List of words representing the input sentence.
    - window (int): Size of the window to capture surrounding context. Default is 3.

    Returns:
    - List of lists: Each inner list contains the surrounding context for the corresponding word in the input sentence.
      The length of each inner list is (2 * window) + 1, with 'PAD' added for padding where necessary.
    """

    X = []

    for i in range(len(sentence)):
        left_pad = max(window - i, 0)
        right_pad = max(window - len(sentence) + i + 1, 0)
        left_idx = window - left_pad
        right_idx = window - right_pad
        row = (
            left_pad * ["PAD"]
            + sentence[i - left_idx : i + 1 + right_idx]
            + right_pad * ["PAD"]
        )
        X.append(row)
        assert len(row) == (2 * window) + 1, f"Length:{len(sentence)}, Row:{row}, i:{i}"

    return X


def get_token_tags(file_path):
    """
    Extracts token-tags pairs from files and returns separate lists for tags & tokens.

    Args:
    - file_path (str): Path to the text files.

    Returns:
    - list: List of tags.
    - list: List of tokens.
    """
    tokens, tags = [], []

    txt_files = [file for file in os.listdir(file_path) if file.endswith(".txt")]

    print("Text Files:", txt_files)

    for file_name in txt_files:
        file_path_full = os.path.join(file_path, file_name)
        tmp_token, tmp_tag = [], []

        with open(file_path_full, "r", encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                tmp_token.append(parts[0])
                tmp_tag.append(parts[1])
            elif parts[-1] == "" and len(tmp_token) > 0:
                tokens.append(tmp_token)
                tags.append(tmp_tag)
                tmp_token, tmp_tag = [], []
            else:
                continue
    return tags, tokens


def average_person_length(tags):
    """
    Calculate the average length of person names in a list of tags.

    Parameters:
    - tags (list): A list of tags containing "B-PER" and "I-PER" tags.

    Returns:
    - float: The average length of person names.

    Raises:
    - ValueError: If the input list is empty or does not contain any "B-PER" tag.
    """
    if not tags:
        raise ValueError("Input list is empty. Cannot calculate average.")

    tags_counter = Counter(tags)
    b_per_count = tags_counter["B-PER"]

    if b_per_count == 0:
        raise ValueError(
            "Input list does not contain any 'B-PER' tag. Cannot calculate average."
        )

    total_length = b_per_count + tags_counter["I-PER"]
    average_length = round(total_length / b_per_count, 2)

    return average_length


def remove_items(tokens, tags):
    """
    Remove items from lists by their index values (in-place).

    Parameters:
    - tokens (list): tokens list.
    - tags (list): tags list.
    """
    idx_to_remove = []
    for idx, tag in enumerate(tags):
        if "B-PER" not in tag and "I-PER" not in tag:
            idx_to_remove.append(idx)

    for index in sorted(idx_to_remove, reverse=True):
        tokens.pop(index)
        tags.pop(index)


def average_tag_presence(tags):
    """
    Calculate the percentage of sentences containing a specific tag in a dataset.

    Parameters:
    - tags (list): List of tags associated with sentences.
    """

    sentence_with_tag = 0
    dataset_size = len(tags)
    for tag in tags:
        if "B-PER" in tag or "I-PER" in tag:
            sentence_with_tag += 1

    percentage = round(sentence_with_tag * 100 / dataset_size, 2)
    print(f"Percentage of sentences containing Peson name: {percentage}%")


def get_vector(token, word_model):
    """
    Get the vector representation of a token from a word embedding model.

    Parameters:
    - token (str): The token for which the vector is requested.
    - word_model (Word2Vec): The word embedding model.

    Returns:
    - numpy.ndarray: The vector representation of the token.
    """
    try:
        return word_model.wv[token]
    except KeyError:
        return np.zeros(
            word_model.vector_size,
        )


def row2vec(row, word_model):
    """
    Convert a list of tokens into a single vector by concatenating their word vectors.

    Parameters:
    - row (list): A list of tokens.
    - word_model (Word2Vec): The word embedding model.

    Returns:
    - numpy.ndarray: The vector representation of the entire row.
    """
    rowvec = [get_vector(token, word_model) for token in row]
    length = word_model.vector_size * len(row)
    return np.array(rowvec).reshape(length)


def random_indices_by_category(category_array, n_sample):
    """
    Randomly selects indices from each unique category in a NumPy array.

    Parameters:
    - category_array (numpy.ndarray): NumPy array containing category labels.
    - n_sample (int): Number of indices to randomly select for each category.

    Returns:
    - List of randomly selected indices, one for each unique category in the input array.
    """

    # Find unique categories and their counts
    unique_categories, _ = np.unique(category_array, return_counts=True)

    # List to store randomly selected indices
    result = []

    # Randomly select indices for each category
    for category in unique_categories:
        indices_for_category = np.where(category_array == category)[0]
        random_indices = np.random.choice(
            indices_for_category,
            size=min(n_sample, len(indices_for_category)),
            replace=False,
        )
        result += random_indices.tolist()

    return result


def get_name(sentence, w2v, model, label2class):
    """
    Extracts person names from a given sentence using a specified word embedding model and a trained classification model.

    Parameters:
    - sentence (str): Input sentence from which person names are to be extracted.
    - w2v: Word embedding model for converting words to vectors.
    - model: Trained classification model for predicting named entity labels.
    - label2class (dict): Mapping of label indices to human-readable class names.

    Returns:
    - Tuple: A tuple containing two elements:
      1. List of extracted person names from the sentence.
      2. List of predicted labels for each word in the sentence.

    Example:
    names, result = get_name("John Doe is a software engineer.", w2v_model, trained_model, label_mapping)
    print("Extracted Names:", names)
    print("Predicted Labels:", result)
    """

    print("Given sentence:", sentence)
    words = sentence.strip().split(" ")
    feature = get_surrounding(words)
    vector = list(map(lambda x: row2vec(x, w2v), feature))
    vector = np.array(vector).reshape(len(vector), -1)

    try:
        result = model.predict(vector)
        result = [label2class[tag] for tag in result]
    except TypeError:
        y_pred = model.predict(vector, verbose=False)
        result = tf.argmax(y_pred, axis=1).numpy().tolist()
        result = [label2class[tag] for tag in result]

    names = []
    print("Extracted names:", end=" ")
    for tag, word in zip(result, words):
        if tag == "B-PER":
            if len(names) > 0:
                names.append(" ")
                print("," + word, end=" ")
            else:
                print(word, end=" ")
            names.append(word)
        elif tag == "I-PER":
            names.append(word)
            print(word, end=" ")
    print()
    return names, result
