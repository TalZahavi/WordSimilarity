import pickle
from nltk.stem import WordNetLemmatizer
import string
from collections import Counter


class Trainer:
    CONTEXT_LIMIT = 5000

    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.context_words = dict()

    # Pre process the data, and get the "limit" most frequent words
    def pre_process(self):
        sentence_number = 0
        word_freq = dict()  # For a given word, return how many times the word appear in the data
        with open('Data\\en\\full.txt', 'r') as f:
            for line in f:
                sentence_number += 1
                # Remove char that are not in english
                english_line = ''.join(filter(lambda x: x in string.printable, line))
                # Remove punctuation
                without_punctuation = english_line.translate(string.maketrans("", ""), string.punctuation)
                # Get the sentence as an array of words
                line_arr = (" ".join(without_punctuation.split())).split(' ')
                for word in line_arr:
                    # Replace numbers\years with special chars
                    if word.isdigit():
                        if len(word) == 4:
                            processed_word = "$YEAR$"
                        else:
                            processed_word = "$NUM$"
                    elif word[:-1].isdigit() and len(word[:-1]) == 4 and word.endswith('s'):
                        processed_word = "$YEAR$"
                    # The word is not a number
                    else:
                        # Un capital word
                        lower_word = word.lower()
                        # Lemmatization
                        processed_word = self.wordnet_lemmatizer.lemmatize(lower_word).encode('utf-8')

                    # Enter the word to the dict
                    if processed_word in word_freq:
                        word_freq[processed_word] += 1
                    else:
                        word_freq[processed_word] = 1

                print('Done pre processing ' + str((float(sentence_number)/float(880836260))*100) + '%')
                if sentence_number == 10000:
                    break

        f.close()
        # Get the most limit words
        common_words = Counter(word_freq).most_common(30)
        print(common_words)


trainer = Trainer()
trainer.pre_process()
