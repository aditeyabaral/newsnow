import warnings
warnings.filterwarnings("ignore")
import nltk, string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
#nltk.download('punkt')
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
def cosine_sim(text1, text2):
    wt1,wt2 = text1.split(),text2.split()
    text1,text2 = '',''
    for i in wt1:
        if i in stop_words:
            wt1.remove(i)
    for i in wt2:
        if i in stop_words:
            wt2.remove(i)
    for i in wt1:
        text1+=i + '. '
    for i in wt2:
        text2+=i + '. '
    text1 = text1.strip()
    text2 = text2.strip()
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]
vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
def merge(content):
    final = ''
    model = max(content, key = len)
    #print(model)
    material = [i for i in content if i!=model]
    total = [model] + material
    lines_in_model = model.split(sep = '.')
    while len(material)>0:
        text = material[0]
        pos = []
        lines_in_text = text.split(sep = '.')
        for text_line in lines_in_text:
            pos = []
            for model_line in lines_in_model:
                try:
                    sim = cosine_sim(model_line.strip(),text_line.strip())
                    pos.append([lines_in_model.index(model_line),sim])
                except:
                    pass
            l = len(pos)
            for i in range(0,l):
                for j in range(0,l-i-1):
                    if pos[j][1]>pos[j+1][1]:
                        pos[j],pos[j+1] = pos[j+1],pos[j]
            pos.reverse()
            insert_position = pos[0][0]
            if insert_position>0.75:
                lines_in_model.insert(insert_position,text_line.strip())
        del material[0]
    for lines in lines_in_model:
        final+=lines.strip()+'. '
    return final.strip()
        
                
        
    

    
