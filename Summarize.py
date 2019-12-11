import warnings
warnings.filterwarnings("ignore")
import imp,nltk,requests,re,heapq
from gensim.summarization import summarize
from bs4 import BeautifulSoup
stopwords = nltk.corpus.stopwords.words('english')
def summ(s):
    sentence_list = nltk.sent_tokenize(s) 
    formatted_article_text = s
    word_frequencies = {}  
    for word in nltk.word_tokenize(formatted_article_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 50:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    summary_sentences = heapq.nlargest(15, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)  
    return summary
