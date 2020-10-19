import re
import string
import heapq
import requests
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from gensim.summarization import summarize as gensim_summarize
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from GoogleNews import GoogleNews


stopwords = stopwords.words("english")
stemmer = SnowballStemmer("english")
vectorizer = TfidfVectorizer(
    stop_words="english",
    decode_error="ignore",
    lowercase=True,
    # ngram_range=(1, 3)
)


def cosine_similarity(A, B):
    return np.dot(A, B)/(np.linalg.norm(A) * np.linalg.norm(B))


def getLinks(query, num_links=5):
    googlenews = GoogleNews(lang="en")
    googlenews.search(query)
    return googlenews.get__links()[:num_links]


def getDocuments(urls):
    articles = list()
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraph_texts = soup.find_all('p')
        content = [re.sub(r'<.+?>', r'', str(p)) for p in paragraph_texts]
        content = [re.sub(
            r"[\"\#\%\&\(\)\*\+\/\:\<\=\>\@\[\\\]\^\_\`\{\|\}\~]+", " ", document) for document in content]
        content = [re.sub(r"[ \t\n\r\x0b\x0c]+", " ", document)
                   for document in content]
        if content:
            articles.append(" ".join(content))
    return articles


def merge(documents, threshold=0.75):
    documents_sentences = list(map(sent_tokenize, documents))
    largest_document = max(documents_sentences, key=len)
    final_document = largest_document
    vectorizer.fit(largest_document)
    for document in documents_sentences:
        if document == largest_document:
            continue
        for document_line_position, document_line in enumerate(document):
            position = list()
            for final_document_line_position, final_document_line in enumerate(final_document):
                document_line_vector, final_document_line_vector = vectorizer.transform(
                    [document_line, final_document_line]).toarray()
                similarity = cosine_similarity(
                    document_line_vector, final_document_line_vector)
                position.append((final_document_line_position, similarity))
            position.sort(reverse=True, key=lambda x: x[1])
            best_position, highest_similarity = position[0]
            if highest_similarity >= threshold:
                final_document.insert(best_position, document_line)
    return " ".join(final_document)


def summarize(corpus, mode='rank', ratio=0.5, num_sentences=15):
    if mode == "frequency":
        sentence_list = sent_tokenize(corpus)
        formatted_article_text = corpus
        word_frequencies = dict()
        for word in word_tokenize(formatted_article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        maximum_frequncy = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

        sentence_scores = dict()
        for sent in sentence_list:
            words = word_tokenize(sent.lower())
            count_words = len(words)
            for word in words:
                if word in word_frequencies.keys():
                    if count_words < 50:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        summary_sentences = heapq.nlargest(
            num_sentences, sentence_scores, key=sentence_scores.get)
        summary = " ".join(summary_sentences)
        return summary
    else:
        return gensim_summarize(corpus, ratio)
