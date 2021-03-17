#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
sys.path.append("/usr/local/lib/python3.9/site-packages")

import nltk, pandas as pd, numpy as np
import re
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree
#nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

from nrclex import NRCLex

#nltk.download('vader_lexicon') 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

import pickle
from fuzzywuzzy import fuzz
#nltk.download('punkt')
from nltk import tokenize
import re
from tqdm import tqdm
tqdm.pandas()
from string import digits
from string import punctuation


# In[5]:


dep_parser = CoreNLPDependencyParser(url='http://0.0.0.0:9000')
pos_tagger = CoreNLPParser(url='http://0.0.0.0:9000', tagtype='pos')


# In[6]:


def convert_sentence (input_sent):
    # Parse sentence using Stanford CoreNLP Parser
    pos_type = pos_tagger.tag(input_sent.split())
    parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
    dep_type, = ParentedTree.convert(dep_parser.parse(input_sent.split()))
    return pos_type, parse_tree, dep_type

def multi_liaison (input_sent, output=['tagging','parse_tree','type_dep','spo','relation']):
    pos_type, parse_tree, dep_type = convert_sentence(input_sent)
    pos_sent = ' '.join([x[0]+'/'+x[1] for x in pos_type])
    # Extract subject, predicate and object
    subject, adjective = get_subject(parse_tree)
    predicate = get_predicate(parse_tree)
    objects = get_object(parse_tree)
    # Generate the relations between subjects and objects
    relation = get_relationship(dep_type, subject, predicate, objects)
    if 'tagging' in output:
        print('---TAGGING---')
        print(pos_sent)
        print()
    if 'parse_tree' in output:
        print('---PARSE TREE---')
        parse_tree.pretty_print()
        print()
    if 'type_dep' in output:
#         print('---TYPED DEPENDENCIES---')
        li = []
        for x in dep_type.triples(): li.append(list(x))
        return li
#         print()
    if 'spo' in output:
        print('---MULTI-LIAISON OUTPUT---')
        print('Subject: ',len(subject))
        for x in subject: print(' '.join(x))
        print('Predicate: ',len(predicate))
        for x in predicate: print(' '.join(x))
        print('Object: ',len(objects))
        for x in objects: print(' '.join(x))
        print()
    if 'relation' in output:
        print('---RELATIONSHIP---')
        for x in relation: print(x)

def get_subject (parse_tree):
    # Extract the nouns and adjectives from NP_subtree which is before the first / main VP_subtree
    subject, adjective = [],[]
    for s in parse_tree:
        if s.label() == 'NP':
            for t in s.subtrees(lambda y: y.label() in ['NN','NNP','NNS','NNPS','PRP']):
                # Avoid empty or repeated values
                if t.pos()[0] not in subject:
                    subject.append(t.pos()[0])
            for t in s.subtrees(lambda y: y.label().startswith('JJ')):
                if t.pos()[0] not in adjective:
                    adjective.append(t.pos()[0])
    return subject, adjective

def get_predicate (parse_tree):
    # Extract the verbs from the VP_subtree
    predicate = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label().startswith('VB')):
            if t.pos()[0] not in predicate:
                predicate.append(t.pos()[0]) 
    return predicate

def get_object (parse_tree):
    # Extract the nouns from VP_NP_subtree
    objects, output = [],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label() == 'NP'):
            for u in t.subtrees(lambda z: z.label() in ['NN','NNP','NNS','NNPS','PRP$']):
                output = u.pos()[0]
                if u.left_sibling() is not None and u.left_sibling().label().startswith('JJ'):
                    output += u.left_sibling().pos()[0]
                if output not in objects:
                    objects.append(output)
    return objects

def get_relationship (dep_type, subject, predicate, objects):
    # Generate relations based on the relationship dependencies obtained from parse_tree.triples()
    subject = [x[0] for x in subject]
    predicate = [x[0] for x in predicate]
    objects = [x[0] for x in objects]     
    d1, d2, r1, r2, relation, s1, s2, subjs = [],[],[],[],[],[],[],[]
    w1, w2, output = '','',''
    for head, rel, dep in dep_type.triples():
        if rel in ['nsubj','acl:relcl','conj']:
            s1, s2 = head[0], dep[0]
            if s2 in subject and s1 in predicate:
                w1, w2 = s2, s1
            elif s2 in predicate and (s1 in subject or s1 in objects):
                w1, w2 = s1, s2
            elif s2 in subject and s1 in subject:
                subjs = [s1, s2]
            if w1 != '' and w2 != '':
                r1 = [w1, w2]
        if rel in ['dobj','prep','nmod','conj']:
            d1, d2 = head[0], dep[0]
            if d1 in objects and d2 in objects: 
                r2 = [d1,d2]
            elif d2 in objects:
                r2 = [d2]
            elif d1 in objects:
                r2 = [d1]
        if len(r1) != 0 and len(r2) != 0 and (r2[0] not in r1 and r2[-1] not in r1):
            if len(subjs) != 0:
                for n in subjs:
                    output = '-'.join([n] + r1[-1:] + r2)
                    if output not in relation:
                        relation.append(output)
            else:
                output = '-'.join(r1+r2)
                if output not in relation:
                    relation.append(output)  
    rm = [x for x in relation for y in relation if x != y and re.match(x,y)]
    for x in rm: 
        if x in relation:
            relation.remove(x)    
    return relation


# ### CoreNLP Test

# In[7]:


multi_liaison("This is a test.", output='type_dep')


# ### Cleaning

# In[8]:


def clean_sentence(narrative):
    li = tokenize.sent_tokenize(narrative)
    cleaned = []
    
    for sentence in li:
        sentence = sentence.lower()
        sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
        sentence = sentence.replace(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|rt|\d+', '')
        sentence = sentence.replace(r'^\s+|\s+$', '') 
        sentence = sentence.replace(r'[^\w]', ' ')
        sentence = sentence.translate(str.maketrans('','',digits))
        sentence = sentence.translate(str.maketrans('', '', punctuation))
        senntence = re.sub(r'[^\w]', ' ', sentence)
        sentence = ' '.join([w for w in sentence.split() if w not in (stopwords)])
        cleaned.append(sentence)
        
    return cleaned


# In[9]:


def clean_paragraph(narrative, column_name):
    df['paragraphs'] = df[column_name].fillna('')
    df['paragraphs'] = df['paragraphs'].str.lower()
    df['paragraphs'] = df['paragraphs'].str.replace(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|rt|\d+', '')
    df['paragraphs'] = df['paragraphs'].str.replace(r'^\s+|\s+$', '') 
    df['paragraphs'] = df['paragraphs'].apply(lambda x: ' '.join([w for w in x.split() if w not in (stopwords)]))
    return df


# ### Import

# In[10]:


infile = open("lexicon", "rb")
lexicon = pickle.load(infile)
infile.close()


# In[11]:


lexicon = list(lexicon)


# ### Behavioral Model

# In[12]:


def get_behavior_breakdown(string):
    li = multi_liaison(string, output="type_dep")
    behaviors = {}
    for group in li:
        if (group[0][1].startswith('JJ') or group[0][1].startswith('VB')) and group[0][0] not in behaviors:
            behaviors[group[0][0]] = []
        if group[2][1].startswith('RB') and group[0][0] in behaviors.keys():
            behaviors[group[0][0]].append(group[2][0])
    return behaviors


# In[13]:


#todo: check if behaviors are in lexicon
def check_behaviors(behaviors, lexicon, threshold):
    
#     dictionary = get_behavior_breakdown(narrative)
    
    for word, modifier in behaviors.items():
        if "not" in modifier:
            continue
        else:
            for behavior in lexicon:
                if word in lexicon:
                    return True
    
    return False


# ### Sentiment Model

# In[14]:


def get_sentiment(string):
    result = sid.polarity_scores(string)
    if (result['compound'] > 0):
        return "POS"
    elif (result['compound'] == 0):
        return "NEU"
    else:
        return "NEG"


# In[15]:


def get_sentiment_breakdown(string):
    text_object = NRCLex(string)
    frequencies = text_object.affect_frequencies
    return frequencies


# ### Checker

# In[16]:


def check(sentiment, behavior):
    if (sentiment == "NEG" or behavior == True):
        return "UNWELL"
    else:
        return "WELL"


# In[17]:


def check_v2(row, narrative, sentences):
   while True: 
        try:
            print(row.name)
            dictionary = {}
            true = 0
            false = 0
            _hasBehavior = False

            for sentence in sentences:
                behaviors = {}
                if (sentence):
                    behaviors = get_behavior_breakdown(sentence)
                hasBehavior = check_behaviors(behaviors, lexicon, 90)

                if hasBehavior == True:
                    true+=1
                else:
                    false+=1

                dictionary.update(behaviors)

            sentiments = get_sentiment_breakdown(narrative)
            sentiment_val = get_sentiment(narrative)

            if true > false:
                _hasBehavior = True
            else:
                _hasBehavior = False

            if (sentiment_val == "NEG" or _hasBehavior == True):
                return pd.Series(["U", dictionary, sentiments])
            else:
                return pd.Series(["W", None, None])
            break;
        except:
            print("Sentence:", sentence)
            print("Error:", row.name)
            return pd.Series([None, None, None])
            break;


# In[ ]:





# In[27]:


sys.modules[__name__] = get_behavior_breakdown


# In[ ]:




