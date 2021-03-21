import pandas as pd 
import string
import nltk
from nltk.corpus import stopwords as nltk_stopwords


class Filter:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')       

    def pre_process(self, sms):
        stopwords = nltk_stopwords.words('english')
        punctuation = string.punctuation
        remove_punct = ''.join([word.lower() for word in sms if word not in punctuation])
        tokenize = nltk.tokenize.word_tokenize(remove_punct)
        remove_stop_words = [word for word in tokenize if word not in stopwords]
        return remove_stop_words
    
    def categorize(self):
        dataset = './datasets/dataset.txt'
        data = pd.read_csv(dataset, sep='\t',  header=None, names=['label', 'sms'])
        data['processed'] = data['sms'].apply(lambda x: self.pre_process(x))

        spam_words = []
        ham_words = []

        for sms in data['processed'][data['label'] == 'spam']:
            for word in sms: spam_words.append(word)

        for sms in data['processed'][data['label'] == 'ham']:
            for word in sms: ham_words.append(word)

        return spam_words, ham_words

    def predict(self, sms):
        spam_words, ham_words = self.categorize()

        spam_count = 0
        ham_count = 0
        result = ''

        for word in sms: 
            spam_count = spam_words.count(word)
            ham_count = ham_words.count(word)

        print(f'{ham_count} ham words found')
        print(f'{spam_count} spam words found')

        if ham_count > spam_count:
            accuracy = round((ham_count / (ham_count + spam_count) * 100))
            print(f'messege is not spam, with {accuracy}% certainty')
            result = f'messege is not spam, with {accuracy}% certainty'
        elif ham_count == spam_count:
            print('message could be spam')
            result = 'message could be spam'
        else:
            accuracy = round((spam_count / (ham_count + spam_count)* 100))
            print(f'message is spam, with {accuracy}% certainty')
            result = f'message is spam, with {accuracy}% certainty'
        
        return result, ham_count, spam_count
