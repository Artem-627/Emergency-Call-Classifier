import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from sklearn import svm
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  
from collections import Counter

max_words = 10000

# Загрузка модели
with open('model.pkl', 'rb') as f:
    lr = pickle.load(f)

# Открытие файла набора данных
train_data = pd.read_csv('dataset.csv', sep=',', index_col='idx')

# Обработка входного текста
def preprocess(text, stop_words, punctuation_marks):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            if token not in stop_words:
                preprocessed_text.append(token)
    return preprocessed_text


# Список знаков пунктуации для исключения из входных данных
punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']

# Список стопслов для исключения из входных данных
stop_words = stopwords.words("english")
stop_words.append("'s")
stop_words.append("send")
stop_words.append("please")
stop_words.append("come")
stop_words.append("help")
stop_words.append("n't")
stop_words.append("saw")

# Обработка текста в датасете
train_data['Preprocessed_texts'] = train_data.apply(lambda row: preprocess(row['Text'], punctuation_marks, stop_words), axis=1)

# Данные для токенизации слов
words = Counter()
for txt in train_data['Preprocessed_texts']:
    words.update(txt)


# Словарь, отображающий слова в коды
word_to_index = dict()
# Словарь, отображающий коды в слова
index_to_word = dict()

# Назначение токенов словам
for i, word in enumerate(words.most_common(max_words - 2)):
    word_to_index[word[0]] = i + 2
    index_to_word[i + 2] = word[0]

# Функция для преобразования списка слов в список кодов
def text_to_sequence(txt, word_to_index):
    seq = []
    for word in txt:
        index = word_to_index.get(word, 1)
        if index != 1:
            seq.append(index)
    return seq

# Функция токенизации данных
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for index in sequence:
            results[i, index] += 1.
    return results


# Создаем сервер
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Корень для проверки
@app.route('/')
def check():
    return "ok"

# Предказание ввода
@app.route('/predict')
def pedict():
    text = request.args.get('text')
    
    positive_preprocessed_text = preprocess(text, stop_words, punctuation_marks)
    positive_seq = text_to_sequence(positive_preprocessed_text, word_to_index)
    positive_bow = vectorize_sequences([positive_seq], max_words)

    result = lr.predict(positive_bow)
    return result[0]

# Main
if __name__ == '__main__':
    app.run(host="90.156.204.192")
