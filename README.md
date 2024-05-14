# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Develop an LSTM-based model capable of accurately recognizing named entities within text data. To train our model, we vectorize these phrases utilizing Embedding layer method.Recurrent neural networks that operate in both directions (bi-directional) and can link two or more hidden layers to a single output.


## DESIGN STEPS

### STEP 1:Load the NER dataset and using Pandas.

### STEP 2:Preprocess the text data and Encode words and labels into numerical format suitable for LSTM input.

### STEP 3:Design the LSTM-based model architecture and Split the dataset into training, validation, and test sets.

### STEP 4:Evaluate the trained model using the test set and


## PROGRAM
``` PYTHON
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)
data = data.fillna(method="ffill")
data.head(50)
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())
print("Unique tags are:", tags)
num_words = len(words)
num_tags = len(tags)
num_words
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
sentences[0]
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
word2idx
plt.hist([len(s) for s in sentences], bins=50)
plt.show()
X1 = [[word2idx[w[0]] for w in s] for s in sentences]
type(X1[0])
X1[0]
max_len = 50
nums = [[1], [2, 3], [4, 5, 6]]
sequence.pad_sequences(nums)
nums = [[1], [2, 3], [4, 5, 6]]
sequence.pad_sequences(nums,maxlen=2)
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
X[0]
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
X_train[0]
y_train[0]
input_word = layers.Input(shape=(max_len,))
embedding_layer= layers.Embedding(input_dim=num_words,output_dim=51,input_length=max_len)(input_word)
dropout_layer=layers.SpatialDropout1D(0.1)(embedding_layer)
bidirectional_lstm=layers.Bidirectional(layers.LSTM(units=102,return_sequences=True,recurrent_dropout=0.1))(dropout_layer)
output=layers.TimeDistributed(layers.Dense(num_tags,activation="softmax"))(bidirectional_lstm)              
model = Model(input_word, output)
model.summary()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=36,
    epochs=2,
)

metrics = pd.DataFrame(model.history.history)
metrics.head()
print("GOWTHAM N")
print("212222220013")
metrics[['accuracy','val_accuracy']].plot()
print("GOWTHAM N")
print("212222220013")
metrics[['loss','val_loss']].plot()

i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

## OUTPUT
![image](https://github.com/krithygowthamn/named-entity-recognition/assets/122247810/f09392de-3a32-4713-bd17-dedc739763c9)

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/krithygowthamn/named-entity-recognition/assets/122247810/4e4225d7-153f-4197-ac7d-5e33bc58d1b5)

![image](https://github.com/krithygowthamn/named-entity-recognition/assets/122247810/2fc55010-c934-4875-ac69-ffa7308ad5a9)

### Sample Text Prediction
![image](https://github.com/krithygowthamn/named-entity-recognition/assets/122247810/e1b11c51-b5d3-4302-a577-c647fe9d2ff4)


## RESULT
Thus, an LSTM-based model (bi-directional) for recognizing the named entities in the text is developed Successfully.
