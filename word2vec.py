from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
nltk.download('punkt') 
import csv
# Sample data
texts = []
with open('your_file.csv', 'r') as file:
    reader = csv.reader(file)

    # Iterate through each line
    for row in reader:
        if row:  # Check if the row is not empty
            user_input = row[0]  # First item in each row before the first comma
            # print(user_input)
            texts.append(user_input)

# Tokenize sentences
tokenized_texts = [word_tokenize(text.lower()) for text in texts]

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Function to get sentence vector by averaging word vectors
def get_sentence_vector(sentence):
    words = word_tokenize(sentence.lower())
    word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

# Convert each sentence to a vector
sentence_vectors = [get_sentence_vector(text) for text in texts]
print(sentence_vectors)
with open('/mnt/d/thon/your_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for vector in sentence_vectors:
        writer.writerow(vector)
