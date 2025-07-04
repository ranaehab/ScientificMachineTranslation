import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, GRU, Concatenate, Dot, Attention, Dense,Embedding,Activation,Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define hyperparameters
max_sequence_length = 100
embedding_dim = 100
latent_dim = 256


# Load train, validate, and test data from text files
with open('Train.txt', 'r', encoding='utf-8') as f:
    train_data = f.readlines()
with open('Test.txt', 'r', encoding='utf-8') as f:
    test_data = f.readlines()

# Extract English and Arabic sentences from data
english_sentences_train = []
arabic_sentences_train = []

english_test = []
arabic_test = []

for line in train_data:
    english, arabic = line.strip().split('\t')
    english_sentences_train.append(english)
    arabic_sentences_train.append(arabic)

for line in test_data:
    english, arabic = line.strip().split('\t')
    english_test.append(english)
    arabic_test.append(arabic)
# Tokenize sentences and convert them into numerical representations
tokenizer_en = Tokenizer()
tokenizer_en.fit_on_texts(english_sentences_train)
train_data_en = tokenizer_en.texts_to_sequences(english_sentences_train)
train_data_en = pad_sequences(train_data_en, maxlen=max_sequence_length, padding='post', truncating='post')
tokenizer_ar = Tokenizer()
tokenizer_ar.fit_on_texts(arabic_sentences_train)
train_data_ar = tokenizer_ar.texts_to_sequences(arabic_sentences_train)
train_data_ar = pad_sequences(train_data_ar, maxlen=max_sequence_length, padding='post', truncating='post')

# Define encoder inputs and GRU layer
encoder_inputs = Input(shape=(max_sequence_length,))
encoder_embedding = Embedding(len(tokenizer_ar.word_index) + 1, embedding_dim)(encoder_inputs)
#CNN Layer
conv1d_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(encoder_embedding)
maxpooling_layer = MaxPooling1D(pool_size=2)(conv1d_layer)

encoder_gru = GRU(latent_dim, return_sequences=True)(maxpooling_layer)
encoder_gru = Dense(latent_dim)(encoder_gru)

# Define decoder inputs and BiLSTM layer
decoder_inputs = Input(shape=(max_sequence_length,))
decoder_embedding = Embedding(len(tokenizer_en.word_index) + 1, embedding_dim)(decoder_inputs)
decoder_bilstm = Bidirectional(LSTM(latent_dim, return_sequences=True))(decoder_embedding)

# Slice only the ‘hidden_dim’ dimensions from the bidirectional output
decoder_bilstm = decoder_bilstm[:, :, latent_dim:]

# Apply Attention mechanism
attention = Dot(axes=[2, 2])([decoder_bilstm, encoder_gru])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, encoder_gru])
decoder_combined_context = Concatenate(axis=-1)([context, decoder_bilstm])

# Define decoder output layer
decoder_dense = Dense(len(tokenizer_en.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.summary()

# Train the model
model.fit([train_data_ar,train_data_en], train_data_en, batch_size=64, epochs=30, validation_split=0.2)

# Use test data to evaluate the model
test_data_en = tokenizer_en.texts_to_sequences(english_test)
test_data_en = pad_sequences(test_data_en, maxlen=max_sequence_length, padding='post', truncating='post')
test_data_ar = tokenizer_ar.texts_to_sequences(arabic_test)
test_data_ar = pad_sequences(test_data_ar, maxlen=max_sequence_length, padding='post', truncating='post')
predicted_arabic = model.predict([test_data_ar,test_data_en])
decoded_arabic = tokenizer_en.sequences_to_texts(np.argmax(predicted_arabic, axis=-1))

# Print example prediction translation from test data
for i in range(len(english_test)): 
    #print("English sentence: ", english_test[i])
    print("Predicted Arabic sentence: ", decoded_arabic[i])
