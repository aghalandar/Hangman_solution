# Hangman challenge

## Question
Trexquant Interview Project (The Hangman Game):
Instruction:
For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server.

When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word or (2) the user has made six incorrect guesses.

You are required to write a "guess" function that takes the current word (with underscores) as input and returns a guess letter. You will use the API codes below to play 1,000 Hangman games. You have the opportunity to practice before you start recording your game results.

Your algorithm is permitted to use a training set of approximately 250,000 dictionary words. Your algorithm will be tested on an entirely disjoint set of 250,000 dictionary words. Please note that this means the words that you will ultimately be tested on do NOT appear in the dictionary that you are given. You are not permitted to use any dictionary other than the training dictionary we provided. This requirement will be strictly enforced by code review.

You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearance that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.

This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.

## Solution 1: using LSTM 
Input:
To use the provided dictionary effectively, an n-gram model is employed as input for the LSTM model. The process involves generating all possible unique n-grams with lengths ranging
from 2 to 6 using words from the dictionary. Next, all permutations are constructed from each n-gram containing at least one underscore ('_'). This results in approximately 30 million
permutations. For consistent input data length (set to 6), any remaining space in n-grams is filled with '*'. The characters (a-z, '_', '*') are mapped to the range (1 - 26), 27, and 28, respectively. 

For instance, for a 2-gram 'ab', the following input is constructed:

Word: ‘ab’

Permutations: [_b], [a_]

Fixed length: [_b****], [a_****]

Encoded Permutations: [27, 2, 28, 28, 28, 28], [1, 27, 28, 28, 28, 28]

#### Output:

An array of length 26 is created for each letter in the alphabet. Each element signifies the presence of the corresponding alphabet in the full n-gram.

#### Strategy:
The prediction is based on a combination of:

1. Frequencies of letters in the current dictionary of potential words. I also use the most frequent letters in words of the same length to establish initial input for the LSTM and n-gram models.

2. Predictions from the LSTM model (bi-LSTM), incorporating unique n-grams derived from the masked word. This combination ensures a comprehensive approach to letter guessing in the Hangman game.

## Solution 2: using Trie structure

#### Approach:
In the second approach, I employ a trie due to its powerful functionality. The trie structure serves as an intelligent way to store words or sequences, enabling fast searches, efficient space utilization by storing only necessary data, and supporting auto-complete features in search engines. Adding or removing words is akin to adjusting branches on a tree, making it versatile and straightforward. Moreover, it excels in sorting numerous words swiftly while maintaining a tidy and compact structure.

This approach for tackling the Hangman challenge combines heuristic techniques with a trie data structure for letter prediction. Initially, we construct a trie to efficiently store and navigate the dictionary, integrating frequency information at each node. During the guessing process, our algorithm looks for patterns that match the masked word within the constructed trie.

#### Strategy:  
The strategy involves several key components:

* Filtering the dictionary to extract words of the same length and pattern as the masked word.

* Predicting potential letters based on observed patterns within the word and leveraging the information stored in the trie.

* Using prefix checks to enhance the accuracy of predictions.

* As a backup, use patterns derived from both the subset of words matching the masked word's length and the entire dictionary for comprehensive coverage.

This strategy combines various methods to optimize letter prediction and enhance overall performance in the Hangman challenge.


