import glob
import re, os, sys
import argparse
import numpy as np
import pandas as pd

from gensim import corpora
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import preprocess_string,remove_stopwords,stem_text
from gensim import similarities
import matplotlib.pyplot as plt


# Create the parser
my_parser = argparse.ArgumentParser(description='Display the similar books to a given book based on the tfidf')

# Add the arguments
my_parser.add_argument('Title',
                       metavar='title',
                       type=str,
                       help='the title of the book')


# List all the .txt files and sort them alphabetically
folder = "datasets/"
files = glob.glob(folder+'*.txt')



# Read the text file and gather book titles
txts = []
titles = []

print('Loading the data...')
for file in files:
    # Open each file
    f = open(file, encoding='utf-8-sig')
    txt = f.read()
    # Remove all non-alpha-numeric characters
    txt = re.sub('[\W_]',' ',txt)
    title = (os.path.basename(file)).replace('.txt','')
    # Store the texts and titles of the books in two separate lists
    txts.append(txt)
    titles.append(title)

#books titles to id mapping
titles2id = {titles[i]:i for i in range(len(titles))}

# lowercasing, removing stopwords and stemming for each book
filter = [lambda t:t.lower(),remove_stopwords,stem_text]
processed_txts = [preprocess_string(t,filter) for t in txts]

#Create vocabulary from processed texts
vocab = corpora.Dictionary(processed_txts)

# Create a bag-of-words model for each book, using the previously generated dictionary
bows = [vocab.doc2bow(t) for t in processed_txts]
tfidf_model= TfidfModel(bows)



def get_tfidf(book_id):
    #Return the tfidf representations for a particular book
    return  tfidf_model[bows[book_id]]

def most_common_token(vocab,book_id,n):
    #Return the most common n tokens in a particular book based on tfidf score
    df_tfidf = pd.DataFrame(tfidf_model[book_id],columns=['token_id','tfidf'])
    df_tfidf['token'] = [vocab[id] for id in df_tfidf.token_id.values]
    most_common_n = df_tfidf.sort_values(by='tfidf',ascending=False)['token'].values[:n]
    return most_common_n

# Compute the similarity matrix (pairwise distance between all books)
sims = similarities.MatrixSimilarity(tfidf_model[bows],num_features=len(vocab))
sim_df = pd.DataFrame(list(sims))
sim_df.columns = titles
sim_df.index = titles



def get_similar_books(title):
    #Return the n most similar books to a particular book
    sim_score = sim_df[title].sort_values()
    sim_score.plot.barh()
    plt.xlabel("Cosine distance")
    plt.title(f"Most similar books to {title}")   
    return plt.savefig('output/similar_books.png')



get_similar_books(sys.argv[1])
