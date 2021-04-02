#### TOKENIZATION is a way to represent words in a way for the computer to process them. With the aim of passing them through a neural network that can understand their meaning ####

import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat'
]

# num_words is the maximum number of words to keep
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
# word index is like the dictionary for the computer after tokenizing
word_index = tokenizer.word_index
print(word_index)
