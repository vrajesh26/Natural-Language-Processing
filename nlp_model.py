
# coding: utf-8

# In[6]:


from wordcloud import WordCloud
from nltk.stem.porter import *
from textblob import TextBlob
import gensim
from gensim import corpora
import numpy as np
import pandas as pd
import os
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from nltk import word_tokenize, pos_tag
from collections import defaultdict
import contractions
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_word=stop
stop_word.extend(['http','https'])
import contractions
import time
import matplotlib
matplotlib.use('Agg')

filepath=os.path.join(os.getcwd(),'static')

class sentiment_final:
    def __init__(self,data,name):
        self.data=data
        self.name=name
    
    def analyse_1(self,name):

        def remove_pattern(input_txt, pattern):
            r=re.findall(pattern, input_txt)
            for i in r:
                input_txt = re.sub(i, '', input_txt)
            return input_txt

        input_data=pd.DataFrame(self.data,columns=[name])
        input_data['tidy_tweet'] = np.vectorize(remove_pattern)(input_data[name], "@[\w]*")
        input_data['text_lowered'] = input_data['tidy_tweet'].apply(lambda x: x.lower())
        input_data['remove_contraction']=input_data['text_lowered'].apply(lambda x:contractions.fix(x))
        input_data['tidy_tweet_punc'] = input_data['remove_contraction'].str.replace("[^a-zA-Z]", " ")
        input_data['remove_space'] = [re.sub(' +', ' ', txt) for txt in input_data['tidy_tweet_punc']]
        input_data['remove_strip']=[i.strip() for i in input_data['remove_space']]
        input_data['tweet_token'] = input_data['remove_strip'].apply(lambda x: x.split())
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        lmtzr = WordNetLemmatizer()
        input_data['token_lemma']=input_data['tweet_token'].apply(lambda x: [lmtzr.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(x)])
        input_data['stop']=input_data['token_lemma'].apply(lambda x: [item for item in x if item not in stop_word])
        input_data['meaningful_wordnet']=input_data['stop'].apply(lambda x: [item for item in x if wn.synsets(item)])
        input_data['tweet_final'] = input_data['meaningful_wordnet'].apply(lambda x: " ".join([i for i in x]))
        return input_data

    def cloud(self):

        input_data=self.analyse_1(self.name)
        all_words = ' '.join([text for text in input_data['tweet_final']])
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
        #cloud_fig=os.path.join(filepath,'cloud_output.png')
        new_graph_name = "cloud" + str(time.time()) + ".png"

        for filename in os.listdir('static/'):
            if filename.startswith('cloud'):  # not to remove other images
                os.remove('static/' + filename)

        #plot.savefig('static/' + new_graph_name)
        wordcloud.to_file('static/' + new_graph_name)
        return new_graph_name


    def sentiment_score(self):

        def analize_sentiment(text):
            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0:
                return 1
            elif analysis.sentiment.polarity == 0:
                return 0
            else:
                return -1
        input_data=self.analyse_1(self.name)
        input_data['Sentiment'] = np.array([analize_sentiment(text) for text in input_data['tweet_final'] ])
        input_data['Sentiment']=input_data['Sentiment'].map({1:"Positive", -1:"Negative", 0:"Neutral"})
        return input_data

    def top_model(self,number_of_topics=2):
        input_data=self.analyse_1(self.name)
        topic_data = input_data['tweet_final'].apply(lambda x: x.split())
        dictionary = corpora.Dictionary(topic_data)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        corpus = [dictionary.doc2bow(text) for text in topic_data]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = number_of_topics, id2word=dictionary, passes=2)
        topics = ldamodel.print_topics(num_words=5)
        
        return topics

