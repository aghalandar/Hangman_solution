import pandas as pd
import numpy as np
import os
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

def read_data(file_path):
    """
    Function to read data from a file.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - The content of the file as a string.
    """
    with open(file_path, "r") as f:
        content = f.read()
    return content

def generate_ngrams(word, n):
    """
    Function to generate all possible n-grams from a given word.

    Parameters:
    - word (str): The input word.
    - n (int): The size of the n-grams.

    Returns:
    - List of n-grams.
    """
    ngrams = [word[i:i+n] for i in range(len(word) - n + 1)]
    return ngrams

def create_ngram_dictionary(df, n):
    """
    Function to create a dictionary of n-gram combinations for each word in the dataset.

    Parameters:
    - df (DataFrame): The input DataFrame containing words.
    - n (int): The size of the n-grams.

    Returns:
    - Dictionary where each word is mapped to a list of encoded n-gram combinations.
    """
    ngram_dictionary = {}

    for word in df[0]:
        all_ngrams_for_word = []

        # Generate n-grams for the word
        ngrams = generate_ngrams(word, n)

        # Encode each n-gram
        encoded_ngrams = [ngram for ngram in ngrams]

        # Append the encoded n-grams to the list
        all_ngrams_for_word += encoded_ngrams

        # Map the original word to the list of encoded n-gram combinations in the dictionary
        ngram_dictionary[word] = all_ngrams_for_word

    return ngram_dictionary

def generate_filled_combinations_for_list(word_list):
    """
    Generates filled combinations for words in a given list by replacing letters with underscores.

    Parameters:
    - word_list (list): List of words for which filled combinations are generated.

    Returns:
    - all_filled_combinations (dict): Dictionary where keys are words, and values are lists of filled combinations.
    """
    all_filled_combinations = {}

    for word in word_list:
        for num_underscores in range(1, len(word)):
            positions = list(range(len(word)))
            pairs = list(combinations(positions, num_underscores))

            filled_combinations = [
                "".join(word[idx] if idx in pair else '_' for idx in range(len(word))) for pair in pairs
            ]

            all_filled_combinations.setdefault(word, []).extend(filled_combinations)

    return all_filled_combinations

def create_char_mapping():
    """
    Creates a character-to-index mapping and an index-to-character mapping for a predefined set of characters.

    Returns:
    - char_to_index (dict): Dictionary mapping characters to their corresponding indices.
    - index_to_char (dict): Dictionary mapping indices to their corresponding characters.
    """
    chars = "abcdefghijklmnopqrstuvwxyz_*"
    char_to_index = {char: i for i, char in enumerate(chars)}
    index_to_char = {i: char for i, char in enumerate(chars)}
    return char_to_index, index_to_char

def encode_input(word):
    """
    Encodes the input word into a numerical vector of fixed length (6).

    Parameters:
    - word (str): The input word to be encoded.

    Returns:
    - word_vector (list): Numerical vector representing the encoded input word.
    """
    char_to_index, _ = create_char_mapping()
    embedding_len = 6
    word_vector = [0] * embedding_len

    for letter_no in range(embedding_len):
        if letter_no < len(word):
            word_vector[letter_no] = char_to_index[word[letter_no]]
        else:
            word_vector[letter_no] = char_to_index['*']

    return word_vector

def encode_output(word):
    """
    Encodes the output word into a numerical vector using a character mapping.

    Parameters:
    - word (str): The output word to be encoded.

    Returns:
    - output_vector (list): Numerical vector representing the encoded output word.
    """
    char_mapping, _ = create_char_mapping()
    output_vector = [0] * 26

    for letter in word:
        output_vector[char_mapping[letter]] = 1

    return output_vector

def encode_dictionary(masked_dictionary):
    """
    Encodes words into numerical vectors for machine learning.

    Parameters:
    - masked_dictionary (dict): A dictionary where keys are output words and values are lists of input words.

    Returns:
    - input_data (list): List containing encoded numerical vectors representing input words.
    - target_data (list): List containing encoded numerical vectors representing corresponding output words.
    """
    target_data = []
    input_data = []

    for output_word, input_words in masked_dictionary.items():
        output_vector = encode_output(output_word)
        input_data.extend([encode_input(input_word) for input_word in input_words])
        target_data.extend([output_vector] * len(input_words))

    return input_data, target_data

def convert_to_tensor(input_data, target_data):
    """
    Converts input and target data to PyTorch tensors.

    Parameters:
    - input_data (list): List containing input data in the form of encoded sequences.
    - target_data (list): List containing target data in the form of encoded sequences.

    Returns:
    - input_tensor (torch.Tensor): PyTorch tensor representing the input data.
    - target_tensor (torch.Tensor): PyTorch tensor representing the target data.
    """
    input_tensor = torch.tensor(input_data, dtype=torch.long)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)
    return input_tensor, target_tensor

def save_input_output_data(input_data, target_data):
    """
    Saves input and target data to text files.

    Parameters:
    - input_data (list): List containing input data.
    - target_data (list): List containing target data.
    """
    with open('input_features.txt', 'w') as fp:
        for item in input_data:
            fp.write("%s\n" % item)
        print('Input data saved successfully.')

    with open('target_features.txt', 'w') as fp:
        for item in target_data:
            fp.write("%s\n" % item)
        print('Target data saved successfully.')

def process_and_prepare_datasets(file_path):
    """
    Processes and prepares datasets for machine learning training.

    Reads data, generates n-grams from 2 to 6, creates n-gram dictionaries, flattens and concatenates n-grams,
    encodes n-grams into numerical vectors, and converts data to PyTorch tensors.

    Parameters:
    - file_path (str): The path to the file containing the data.

    Returns:
    - input_tensor (torch.Tensor): PyTorch tensor representing the input data.
    - target_tensor (torch.Tensor): PyTorch tensor representing the target data.
    """
    content = read_data(file_path)
    words = pd.DataFrame(content.split('\n'))

    input_data = []
    target_data = []

    for ngram in range(2, 7):
        ngram_dictionary = create_ngram_dictionary(words, ngram)
        ngram_list = list(set(perm for perms_list in ngram_dictionary.values() for perm in perms_list))
        all_permutations = generate_filled_combinations_for_list(ngram_list)
        current_input_data, current_target_data = encode_dictionary(all_permutations)
        input_data.extend(current_input_data)
        target_data.extend(current_target_data)
        print(f'{ngram}-gram is Done!')

    save_input_output_data(input_data, target_data)
    input_tensor, target_tensor = convert_to_tensor(input_data, target_data)
    print(input_tensor.size(), target_tensor.size())

    return input_tensor, target_tensor

# Example usage
file_path = os.path.join(os.getcwd(), "words_250000_train.txt")
input_tensor, target_tensor = process_and_prepare_datasets(file_path)
