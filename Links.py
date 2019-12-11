import warnings
warnings.filterwarnings("ignore")
import imp,nltk,requests,re,math,Summarize
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
stemmer = nltk.stem.porter.PorterStemmer()
#vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
import MERGENEW
from gensim.summarization import summarize
from bs4 import BeautifulSoup
from googlesearch import search
import sys
import datetime,time
def get_links():
    query = input("Enter query to search : ")
    time = datetime.datetime.now()
    date,time = str(time).split()
    new_date = date.split('-')
    new_date.reverse()
    s = ''
    for i in new_date:
        s+=i+'-'
    date = s[:len(s)-1]
    fw = open("Search History.txt","a")
    fw.write(date + '\t' + time[:8] + '\t' + query + '\n')
    fw.close()
    print()
    results =  list(search(str(query)+' news', tld="co.in", num=5, stop=1, pause=2))
    content = []
    for url in results:
        resp=requests.get(url) 
        news_soup = BeautifulSoup(resp.text, "html.parser")
        a_text = news_soup.find_all('p')
        y=[re.sub(r'<.+?>',r'',str(a)) for a in a_text]
        s = ''
        for i in y:
            if '{' in i or '[' in i or '(' in i or '@' in i or '|' in i or '..' in i or '/' in i or '\\' in i or ':' in i:
                pass
            else:
                s+=i
        if len(s)>0:
            content.append(s)
    #print(content)
    #<Try including content editing here>
    print("Latest on", query,':')
    print()
    merged_content = MERGENEW.merge(content)
    summary = Summarize.summ(merged_content)
    return (summary,merged_content)

