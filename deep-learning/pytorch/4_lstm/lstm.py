"""
problem tanımı : lstm ile metin türetme

"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter # kelimeler arası frekans hesaplama
from itertools import product # grid search
import pandas as pd
import numpy as np


# data loading and preprocessing
# product comment
text = """
    "Bu ürün beklentimi karşılamadı.
    "Malzeme kalitesi oldukça düşük ve dayanıksız.
    "Kullanım sırasında birçok sorunla karşılaştım ve memnun değilim!
    "Açıklamalarda belirtilen özelliklerin bazıları eksikti.
    "Bu fiyata daha kaliteli ürünler bulunabilir. """
print(text)
print("------------------")
words = text.replace(".","").replace("!","").lower().split()
print(words)
print("------------------")
word_counts = Counter(words)
vocab = sorted(word_counts,key=word_counts.get, reverse = True) # kelime frekansını büyükten küçüğe sırala
print(vocab)
word_to_idx = {word: i for i,word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# train data
data = [(words[i],words[i+1]) for i in range(len(words)-1)]
print(data)
# create lstm model
class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,):
        super().__init__()
        self.embedding =  nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,vocab_size)

    
    def forward(self,x):
        """
            input -> embedding -> LSTM -> fc -> output

        """
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x.view(1,1,-1))
        output = self.fc(lstm_out.view(1,-1))
        return output

# hyperparameter tuning

# word -> tensor
def prepare_sequence(seq,to_idx):
    return torch.tensor([to_idx[w] for w in seq], dtype=torch.long)


embedding_sizes = [8,16]
hidden_sizes = [32,64]
learning_rates = [0.01,0.003]
best_loss = float("inf") # en kçük kayıp değerini saklamak için bir değişken
best_params = {} # en iyi parametreleri saklamak için bos bir dict

print("Hyperparameter tuning starting")
for emb_size, hidden_size, lr in product(embedding_sizes,hidden_sizes,learning_rates):
    print(f"Deneme: Embedding: {emb_size}, hidden: {hidden_size}, lr : {lr}")
    model = LSTM(len(vocab),embedding_dim=emb_size,hidden_dim=hidden_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    epochs = 50
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for word,next_word in data:
            model.zero_grad()
            intput_tensor = prepare_sequence([word],word_to_idx)
            target_tensor = prepare_sequence([next_word],word_to_idx)
            output = model(intput_tensor)
            loss = loss_function(output,target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch%10 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.5f}")
        total_loss = epoch_loss

    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_dim": emb_size,"hidden_dim":hidden_size,"Learning rate": lr}
        print()
print(f"Best params: {best_params}")

# train with good hyperparameter
"""
Best params: {'embedding_dim': 16, 'hidden_dim': 64, 'Learning rate': 0.003}

"""
final_model = LSTM(vocab_size= len(vocab),embedding_dim= best_params['embedding_dim'],hidden_dim = best_params["hidden_dim"])
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(),lr=lr)

print("Final model training")
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0
    for word, next_word in data:
        final_model.zero_grad()
        intput_tensor = prepare_sequence([word],word_to_idx)
        target_tensor = prepare_sequence([next_word],word_to_idx)
        output = final_model(intput_tensor)
        loss = loss_function(output,target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Final model epoch : {epoch}, Loss: {epoch_loss:.5f}")

# test
# word prediction func : input word and predict n word
def predict_sequence(start_word,num_words):
    current_word = start_word
    output_sequence = [current_word]

    for _ in range(num_words):
        with torch.no_grad():
            # word -> tensor
            intput_tensor =  prepare_sequence([current_word],word_to_idx)
            output = final_model(intput_tensor)
            predicted_idx = torch.argmax(output).item()
            predicted_word = idx_to_word[predicted_idx]
            output_sequence.append(predicted_word)
            current_word = predicted_word
    return output_sequence # tahmin edilen kelime dizisi

start_word = "ürün"
num_predictions = 8
predicted_sequence = predict_sequence(start_word=start_word,num_words=num_predictions)
print(" ".join(predicted_sequence))