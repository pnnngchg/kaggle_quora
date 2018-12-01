#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing import text
from model import *

train_df = pd.read_csv('/home/pczero/Desktop/kaggle_quora/train.csv')
# train_df["question_text"] = train_df["question_text"].map(lambda x: clean_text(x))

test_df = pd.read_csv('/home/pczero/Desktop/data/kaggle_quora/test.csv')
# test_df["question_text"] = test_df["question_text"].map(lambda x: clean_text(x))

X_train = train_df["question_text"].fillna("na").values
X_test = test_df["question_text"].fillna("na").values
y = train_df["target"]


maxlen = 70
max_features = 50000  # 最大单词数，词典的大小
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)  # 最大单词数，词典的大小
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)


lstm = lstm_model(tokenizer)
lstm.summary()

batch_size = 2048
epochs = 3

early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
model_checkpoint = ModelCheckpoint('./lstm.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

hist = lstm.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True)
lstm.save('./lstm.h5')

pred_val_y_2 = lstm.predict([X_val], batch_size=1024, verbose=1)
thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = metrics.f1_score(y_val, (pred_val_y_2 > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))

thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh_2 = thresholds[0][0]
print("Best threshold: ", best_thresh_2)

y_pred_2 = lstm.predict(x_test, batch_size=1024, verbose=True)