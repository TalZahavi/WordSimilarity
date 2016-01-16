import pickle
from nltk.stem import WordNetLemmatizer
import string
from collections import Counter
import csv


class Trainer:
    CONTEXT_LIMIT = 5000

    def __init__(self):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.context_words = dict()
        self.words = set()
        self.freq1_matrix = dict()
        self.freq2_matrix = dict()

    # Need to run this only one time - to get the words from the CSV files
    def get_words_from_csv(self, file_name):
        with open(file_name, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            row_num = 1
            for row in reader:
                if row_num != 1:
                    self.words.add(self.wordnet_lemmatizer.lemmatize(row[0].lower()))
                    self.words.add(self.wordnet_lemmatizer.lemmatize(row[1].lower()))
                row_num += 1

    # TODO: PUT ALSO IN PRE PROCESSING!
    # Replacing numbers with special chars, un capital words and lemmatization
    def word_processing(self, word):
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
        return processed_word

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

                print('Done pre processing ' + str((float(sentence_number)/float(88083626))*100) + '%')

        f.close()
        # Get the most limit words
        common_words = Counter(word_freq).most_common(self.CONTEXT_LIMIT)
        pickle.dump(common_words, open("Results\\context.p", "wb"), protocol=2)

    @staticmethod
    def update_matrix(matrix_dict, word, context_word):
        if word in matrix_dict:
            if context_word in matrix_dict[word]:
                matrix_dict[word][context_word] += 1
            else:
                matrix_dict[word][context_word] = 1
        else:
            matrix_dict[word] = {context_word: 1}

    def make_freq_matrix(self):
        sentence_number = 0
        self.words = pickle.load(open("Results\\words.p", "rb"))
        self.context_words = dict(pickle.load(open("Results\\context.p", "rb")))
        with open('Data\\en\\full.txt', 'r') as f:
            for line in f:
                sentence_number += 1
                # Remove char that are not in english
                english_line = ''.join(filter(lambda x: x in string.printable, line))
                # Remove punctuation
                without_punctuation = english_line.translate(string.maketrans("", ""), string.punctuation)
                # Get the sentence as an array of words
                line_arr = (" ".join(without_punctuation.split())).split(' ')
                for i, word in enumerate(line_arr):
                    word = self.word_processing(word)
                    if word in self.words:  # WE CHECK ONLY WORDS THAT ARE IN THE CSV FILES!
                        list_last_index = len(line_arr)-1
                        for window_index in range(1, 3):
                            # Check right window
                            if i+window_index <= list_last_index:
                                cnx_word = self.word_processing(line_arr[i+window_index])
                                if cnx_word in self.context_words:
                                    self.update_matrix(self.freq2_matrix, word, cnx_word)
                                    if window_index == 1:
                                        self.update_matrix(self.freq1_matrix, word, cnx_word)
                            # Check left window
                            if i-window_index >= 0:
                                cnx_word = self.word_processing(line_arr[i-window_index])
                                if cnx_word in self.context_words:
                                    self.update_matrix(self.freq2_matrix, word, cnx_word)
                                    if window_index == 1:
                                        self.update_matrix(self.freq1_matrix, word, cnx_word)

                print('Done ' + str((float(sentence_number)/float(88083626))*100) + '% of the corpus')
        pickle.dump(self.freq1_matrix, open("Results\\freq1.p", "wb"), protocol=2)
        pickle.dump(self.freq2_matrix, open("Results\\freq2.p", "wb"), protocol=2)


trainer = Trainer()

