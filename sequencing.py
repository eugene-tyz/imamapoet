#### SEQUENCING is similar to TOKENIZATION, except now its representing sentences, taking the tokenized words, it sequences the tokens into an array according to the sentences ####
import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# oov_token is meant to indicate Out Of Vocabulary words, they are words that are not in the corpus (the reference array of sentences)
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

# print(word_index)
# print(sequences)

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_seq)
