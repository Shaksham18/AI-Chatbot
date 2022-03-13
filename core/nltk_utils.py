import numpy as np
import nltk
nltk.download("punkt")
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


class NltkUtils:
    def __init__(self):
        pass

    def tokenize(self, message):
        return nltk.word_tokenize(message)

    def stem(self, word):
        return stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence, all_words):
        tokenized_sentence = [self.stem(word) for word in tokenized_sentence]
        bag = np.zeros(len(all_words), dtype=np.float32)

        for index, word in enumerate(all_words):
            if word in tokenized_sentence:
                bag[index] = 1.0

        return bag
