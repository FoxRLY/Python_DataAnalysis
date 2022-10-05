import warnings
warnings.filterwarnings('ignore')
import re
from collections import defaultdict
import pandas as pd
import math


dataset = pd.read_csv("spam.csv", encoding="ISO-8859-15")
df = dataset.head(1000)
dt = dataset.tail(1000)


def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z]+", message)
    return set(all_words)


def count_words(data):
    counts = defaultdict(lambda: [0,0])
    for index, row in data.iterrows():
        for word in tokenize(row['v2']):
            counts[word][0 if row['v1'] == 'spam' else 1] += 1
    return counts


freq = count_words(df)
all_spam = df['v1'].loc[df['v1'] == "spam"].count()
all_not_spam = df['v1'].loc[df['v1'] == "ham"].count()

def prob(freq, all_spam, all_not_spam):
    return [(word,
             (frequency[0]) / (all_spam),
             (frequency[1]) / (all_not_spam))
            for word, frequency in freq.items()]

def prob_smooth(freq, all_spam, all_not_spam, k = 0.1):
    return [ (word,
            (frequency[0] + k) / (all_spam + 2*k),
            (frequency[1] + k) / (all_not_spam + 2*k))
            for word, frequency in freq.items()]

result = prob(freq, all_spam, all_not_spam)
result_smooth = prob_smooth(freq, all_spam, all_not_spam, k=0.1)

def naive_filter(word_probs, message):
    message_words = tokenize(message)
    spam_prob = not_spam_prob = 0
    for word, prob_if_spam, prob_if_not_spam in word_probs:
        if word in message_words:
            spam_prob *= prob_if_spam
            not_spam_prob *= prob_if_not_spam
        else:
            spam_prob *= (1.0 - prob_if_spam)
            not_spam_prob *= (1.0 - prob_if_not_spam)
    return spam_prob/(spam_prob+not_spam_prob)

def naive_filter_smooth(word_probs, message, k = 0.1):
    message_words = tokenize(message)
    spam_prob = not_spam_prob = 0.0
    for word, prob_if_spam, prob_if_not_spam in word_probs:
        if word in message_words:
            spam_prob += math.log(prob_if_spam)
            not_spam_prob += math.log(prob_if_not_spam)
        else:
            spam_prob += math.log(1.0 - prob_if_spam)
            not_spam_prob += math.log(1.0 - prob_if_not_spam)
    e_spam_prob = math.exp(spam_prob)
    e_not_spam_prob = math.exp(not_spam_prob)
    return e_spam_prob/(e_spam_prob+e_not_spam_prob)


dt_frame = pd.DataFrame()
for index, row in dt.iterrows():
    dt_frame = dt_frame.append({'text': row['v2'], 'prob': naive_filter(result, row['v2']), 'spam': row['v1']}, ignore_index=True)

dt_frame_smooth = pd.DataFrame()
for index, row in dt.iterrows():
    dt_frame_smooth = dt_frame_smooth.append({'text': row['v2'], 'prob': naive_filter_smooth(result_smooth, row['v2']), 'spam': row['v1']}, ignore_index=True)


def func(row):
    if row['prob'] > 0.5:
        return True
    else:
        return False


dt_frame['spam_final'] = dt_frame.apply(func, axis = 1)
dt_frame_smooth['spam_final'] = dt_frame_smooth.apply(func, axis = 1)
good_prediction = 0
good_prediction_smooth = 0
for i in range(1000):
    if dt_frame["spam"][i] == "spam" and dt_frame["spam_final"][i]:
        good_prediction += 1
    elif dt_frame["spam"][i] == "ham" and not dt_frame["spam_final"][i]:
        good_prediction += 1

    if dt_frame_smooth["spam"][i] == "spam" and dt_frame_smooth["spam_final"][i]:
        good_prediction_smooth += 1
    elif dt_frame_smooth["spam"][i] == "ham" and not dt_frame_smooth["spam_final"][i]:
        good_prediction_smooth += 1
print(f"Точность: {good_prediction/float(1000) * 100}")
print(f"Точность со сглаживанием: {good_prediction_smooth/float(1000) * 100}")




