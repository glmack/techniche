#!/usr/bin/env python
# coding: utf-8

# ## Topic model

# In[1]:


import pandas as pd
import numpy as np

from test_model import (get_patent_fields_list, get_ml_patents, 
                        create_title_abstract_col,trim_data, 
                        structure_dataframe, partition_dataframe, 
                        build_pipeline, process_docs, pat_inv_map, get_topics)
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary, mmcorpus
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models import AuthorTopicModel
from gensim.test.utils import common_dictionary, datapath, temporary_file
from smart_open import smart_open

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, punkt, RegexpTokenizer, wordpunct_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer

import json
from pandas.io.json import json_normalize
import requests
import re
import os
import calendar
import requests
from bs4 import BeautifulSoup
import pickle
import warnings

import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim

from pprint import pprint

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# suppress gensim deprecation warnings
warnings.simplefilter('ignore')


# In[3]:


np.random.seed(3)


# ### Acquire data

# In[4]:


get_ipython().run_cell_magic('capture', '', '# supress jupyter notebook output of json response\n\n# acquire ML patent dataset from Patentsview API query\nraw_data_1000 = get_ml_patents(pats_per_page=1000)\nraw_data_2000 = get_ml_patents(pats_per_page=2000)')


# In[5]:


# update path below with user's local path to save pickled raw data
path_raw_data_1000 = '/Users/lee/Documents/techniche/techniche/data/raw_data_1000'
path_raw_data_2000 = '/Users/lee/Documents/techniche/techniche/data/raw_data_2000'

# pickle raw_data_1000 and raw_data_2000
with open(path_raw_data_1000, 'wb') as f:
    pickle.dump(raw_data_1000, f)   
with open(path_raw_data_2000, 'wb') as f:
    pickle.dump(raw_data_2000, f)

# un-comment to de-serialize pickled list of dictionaries
# with open(path_raw_data_1000, 'rb') as f:
#     raw_data_1000 = pickle.load(f) 
# with open(path_raw_data_2000, 'rb') as f:
#     raw_data_2000 = pickle.load(f)

# un-comment to view first dictionary in raw_data_1000
# raw_data_1000[0]


# #### Acquire data - Structure data - 1000 patent documents

# In[6]:


# define keys as criteria to subset dataset
retained_keys = ['patent_number', 'patent_date', 'patent_title',
                 'patent_abstract', 'inventors']


# In[7]:


# subset JSON dict dataset of full api response by keys
data_1000 = trim_data(data=raw_data_1000, keys=retained_keys)

# create item in dict by concatenating patent_title and patent_abstract
data_1000 = create_title_abstract_col(data=data_1000)

# convert dataframe from subsetted dict, organize columns, sort by patent_date
df_1000 = structure_dataframe(data=data_1000)

# partition df_1000 into train and test dataframes
data_train_1000, data_test_1000 = partition_dataframe(df_1000, .8)

# convert dataframes (full, train, test) to list format required by model
text_data_1000 = df_1000.patent_title_abstract.tolist()
text_train_1000 = data_train_1000.patent_title_abstract.tolist()
text_test_1000 = data_test_1000.patent_title_abstract.tolist()

# convert text target in JSON response to list w/o dataframe step
text_list_1000 = []
for i in data_1000:
    text_list_1000.append(i['patent_title_abstract'])


# #### Acquire data - Structure data - 2000 patent documents

# In[8]:


# subset dataset of full api response by keys
data_2000 = trim_data(data=raw_data_2000, keys=retained_keys)

# create item by concatenating patent_title and patent_abstract
data_2000 = create_title_abstract_col(data=data_2000)

# create dataframe, organize columns and sort by patent_date
df_2000 = structure_dataframe(data=data_2000)

# partition dataframe
data_train_2000, data_test_2000 = partition_dataframe(df_2000, .8)

# convert dataframe to list format required by model
text_data_2000 = df_2000.patent_title_abstract.tolist()
text_train_2000 = data_train_2000.patent_title_abstract.tolist()
text_test_2000 = data_test_2000.patent_title_abstract.tolist()

# convert text target in JSON response to list w/o dataframe step
text_list_2000 = []
for i in data_2000:
    text_list_2000.append(i['patent_title_abstract'])


# ### Pre-process text data

# In[9]:


# uncomment to download standard stop words from Spacy
# !python -m spacy download en

# update path with location to save stopwords
path_stopwords = '/Users/lee/Documents/techniche/techniche/data/stopwords/english'
stop_words = stopwords.words(path_stopwords)

# create text pre-processing pipeline to tokenize, clean and lower text
nlp = build_pipeline()

# pre-process documents via json-to-df-to-list workflow above
processed_docs_1000train = process_docs(text_train_1000)

# optional pre-process documents via json-to-list workflow above
# processed_docs_1 = process_docs(text_list)


# In[10]:


### Build corpus and dictionary

# build dictionary
id_to_word_1000train = Dictionary(processed_docs_1000train)

# update path with location to save pickled dictionary
path_pickle_id_to_word = '/Users/lee/Documents/techniche/techniche/data/id_to_word_1000train.pkl'

# pickle dictionary
pickle.dump(id_to_word_1000train, open(path_pickle_id_to_word,'wb'))

# apply term-doc freq (list of (token_id, token_count) tuples) to docs
corpus_1000train = [id_to_word_1000train.doc2bow(doc) for doc in processed_docs_1000train]

# uncomment below to create/view formatted corpus
# formatted_corpus_1000 = [[(id_to_word[id], freq) for id, freq in text] for text in corpus_1000train]
# formatted_corpus_1000
# id_to_word_1000train.token2id


# ## Train model #1: Genism LDA model
# Model #1: implementation: Gensim LDAmodel; k_topics=5; n_docs=1000, partition = 80/20

# In[11]:


# construct model #1
model_1 = LdaModel(corpus=corpus_1000train,
                   id2word=id_to_word_1000train,
                   num_topics=5, 
                   random_state=100,
                   update_every=1,
                   chunksize=100,
                   passes=10,
                   alpha='auto',
                   per_word_topics=True)


# ### Model #1 - Explore and visualize

# In[25]:


# explore topics visually
pyLDAvis.enable_notebook()
viz_topics_model_1 = pyLDAvis.gensim.prepare(model_1, 
                                             corpus_1000train, 
                                             id_to_word_1000train)

# uncomment to view visualization
viz_topics_model_1


# In[13]:


# uncomment to view keywords in n topics in corpus
# pprint(model_1.print_topics())


# In[14]:


# uncomment to view important keywords/weight in topic with idx 0
# pprint(model_1.print_topic(4))


# ### Model #1 - Evaluate
# Evaluate models using coherence and perplexity metrics. As unsupervised learning task, no labels with which to evaluate the "expected" prediction. There is an open research agenda on LDA evaluation approaches (intrinsic vs extrinsic; machine vs human-interpretable, etc., task-specific). 

# #### Model #1 - Evaluate - Pre-process test set

# In[15]:


# pre-process 1000 patents from df-to-list worfklow above
processed_docs_1000test = process_docs(text_test_1000)

# build dictionary with dataset of 1000 patents
id_to_word_1000test = Dictionary(processed_docs_1000test)

# apply term-doc frequency (list of (token_id, token_count) tuples) to 1000 patents
corpus_1000test = [id_to_word_1000test.doc2bow(doc) for doc in processed_docs_1000test]


# #### Model #1 - Evaluate - Coherence
# Calculate topic coherence for topic models with 4-step coherence pipeline (segmentation, probability estimation, confirmation measure, aggregation) from Roeder et al., 2015. "Exploring the space of topic coherence measures", WSDM '15 Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM) 2015, 399-408.

# In[16]:


# calculate coherence metric for train set
coherence_model_1train = CoherenceModel(model=model_1, 
                                        texts=processed_docs_1000train,
                                        dictionary=id_to_word_1000train,
                                        coherence='c_v')
coherence_model_1train_get = coherence_model_1train.get_coherence()
print(coherence_model_1train_get)


# In[17]:


# calculate coherence metric for test_set
coherence_model_1test = CoherenceModel(model=model_1, 
                                       texts=processed_docs_1000test, 
                                       dictionary=id_to_word_1000test, 
                                       coherence='c_v')
coherence_model_1test_get = coherence_model_1test.get_coherence()
print(coherence_model_1test_get)


# In[18]:


# calculate coherence metric for each of n topics in test set
coherence_model_1_per_topic = coherence_model_1test.get_coherence_per_topic()

# uncomment to print coherence_model_1_per_topic
# print(coherence_model_1_per_topic)


# #### Model #1 - Evaluate - Perplexity
# Calculate perplexity metric. Metric calculates and returns per-word likelihood bound using a chunk of documents as evaluation corpus. Output calculated statistics, including the perplexity=2^(-bound), to log at INFO level. Returns the variational bound score calculated for each word

# In[19]:


# calculate perplexity metric for model_1 train set (1000 pats dataset)
perplexity_model_1train = model_1.log_perplexity(corpus_1000train)
print(perplexity_model_1train)


# In[20]:


# calculate perplexity metric for model_1 test set (1000 pats dataset)
perplexity_model_1test = model_1.log_perplexity(corpus_1000test)
print(perplexity_model_1test)


# ### Model #1 - Predict

# #### Model #1 - Predict - Pickle model

# In[21]:


# update path with location to save pickled model
path_pickle_model_1 = '/Users/lee/Documents/techniche/techniche/data/model_lda_1.pkl'

# save/pickle model #1 for subsequent use
pickle.dump(model_1, open(path_pickle_model_1,'wb'))

# uncommment below to load pickled model #1
# model_1 = pickle.load(open(path_pickle_model_1


# #### Model #1 - inference

# In[22]:


## Test model #1 on 2 new text w/ two strings

# define example text_input #1, expressing keyword-type search
text_input_1 = 'smart assistant transformer model translation'.split()

# define example text_input #1, expressing technical details in job post
text_input_2 = """At the Siri International team within Apple we bring the
Siri intelligent assistant to our customers worldwide in over 40 languages
and dialects. Join us, and tackle some of the most challenging problems in
natural language processing and large scale applied machine learning. You 
will build cutting edge natural language understanding technologies and 
deploy them on a global scale. Your work will advance and shape the future
vision of our multi-lingual, multi-cultural Siri assistant, and Search 
applications used by millions across the world Key Qualifications Extensive
track record of scientific research in NLP and Machine Learning, or similar
experience in developing language technologies for shipping products.
Strong coding and software engineering skills in a mainstream programming 
language, such as Python, Java, C/C++. Familiarity with NLP/ML tools and 
packages like Caffe, pyTorch, TensorFlow, Weka, scikit-learn, nltk, etc.
Practical experience building production quality applications related to 
natural language processing and machine learning. In-depth knowledge of 
machine learning algorithms and ability to apply them in data driven natural
language processing systems. Ability to quickly prototype ideas / solutions,
perform critical analysis, and use creative approaches for solving complex 
problems. Attention to detail and excellent communication skills. Description
We are looking for a highly motivated technologist with a strong background 
in Natural Language Processing and Machine Learning research. The ideal 
candidate will have a strong track record of taking research ideas to 
real-world applications. In this position you will apply your problem solving
skills to challenges and opportunities within Siri International, which 
involves development of large-scale language technologies for many natural
languages worldwide. The primary responsibility of this role is to conduct
research and develop innovative machine learning, artificial intelligence 
and NLP solutions for multi-lingual conversational agents. You will have 
the opportunity to investigate cutting edge research methods that will 
improve customer experience of our products and enable our engineers to 
scale these technologies across a variety of natural languages. You will 
also provide technical leadership and experiment-driven insights for 
engineering teams on their machine learning modeling and data decisions. 
You will play a central role in defining the future technical directions 
of Siri International through quick prototyping, critical analysis and 
development of core multi-lingual NLP technologies. Education & Experience
PhD in Machine Learning, Statistics, Computer Science, Mathematics or 
related field with specialization in natural language processing and/or 
machine learning, OR * Masters degree in a related field with a strong 
academic/industrial track record. * Hands-on research experience in an 
academic or industrial setting.""".split()


# In[23]:


get_ipython().run_cell_magic('capture', '', '# suppress cell output in notebook\n\n# pass text through pre-process pipeline\nid_to_word_1000train.doc2bow(text_input_1)\nid_to_word_1000train.doc2bow(text_input_2)')


# In[24]:


predict_input_1 = get_topics(id_to_word_1000train.doc2bow(text_input_1), model_1, k=10)
# uncomment below to view predict_input_1
# predict_input_1


# In[29]:


predict_input_2 = get_topics(id_to_word_1000train.doc2bow(text_input_2), model_1, k=10)
# uncomment below to view predict_input_2
# predict_input_2


# ## Train model #2: Genism LDAMallet model
# Model #2: implementation: Gensim LDAMallet wrapper around LDA Mallet model; 
#           k_topics=5; 
#           n_docs=1000; 
#           partition = 80/20

# In[30]:


# uncomment to download Mallet topic model
# !wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip

# update path with location of saved Mallet topic model
path_mallet = '/Users/lee/Documents/techniche/techniche/data/mallet-2.0.8/bin/mallet'


# In[31]:


# construct model #2
model_2 = gensim.models.wrappers.LdaMallet(path_mallet, 
                                           corpus=corpus_1000train, 
                                           num_topics=5, 
                                           id2word=id_to_word_1000train)


# ### Model #2- Evaluate

# #### Model #2 - Evaluate - Coherence

# In[32]:


# calculate coherence metric for train set (1000 pats dataset)
coherence_model_2train = CoherenceModel(model=model_2, 
                                        texts=processed_docs_1000train,
                                        dictionary=id_to_word_1000train,
                                        coherence='c_v')
coherence_model_2train_get = coherence_model_2train.get_coherence()
print(coherence_model_2train_get)


# In[33]:


# calculate coherence metric for test_set (1000 pats dataset)
coherence_model_2test = CoherenceModel(model=model_2, 
                                       texts=processed_docs_1000test, 
                                       dictionary=id_to_word_1000test, 
                                       coherence='c_v')
coherence_model_2test_get = coherence_model_2test.get_coherence()
print(coherence_model_2test_get)


# In[34]:


# calculate coherence metric for each of n topics in test set
coherence_model_2_per_topic = coherence_model_2test.get_coherence_per_topic()
# print(coherence_model_2_per_topic)


# ## Train Model #3: Gensim LDA model
# Model #3: implementation: Gensim LDAmodel; k_topics=10; n_docs=1000, partition = 80/20
# This model increases the k_topics from 5 to 10, relative to model #1 above

# In[35]:


# construct model #3
model_3 = LdaModel(corpus=corpus_1000train,
                   id2word=id_to_word_1000train,
                   num_topics=10, 
                   random_state=100,
                   update_every=1,
                   chunksize=100,
                   passes=10,
                   alpha='auto',
                   per_word_topics=True)


# ### Model #3 - Explore and visualize

# In[36]:


# explore topics visually
pyLDAvis.enable_notebook()
viz_topics_model_3 = pyLDAvis.gensim.prepare(model_3, 
                                             corpus_1000train, 
                                             id_to_word_1000train)
# viz_topics_model_3


# In[37]:


# keywords in n topics in corpus
# uncomment below to view
# pprint(model_3.print_topics())


# In[38]:


# view most important keywords/weight of topic with idx 0
# uncomment below to view
# pprint(model_3.print_topic(4))


# ### Model #3 - Evaluate

# #### Model #3 - Evaluate - Coherence

# In[39]:


# calculate coherence metric for train set (1000 pats dataset)
coherence_model_3train = CoherenceModel(model=model_3, 
                                        texts=processed_docs_1000train,
                                        dictionary=id_to_word_1000train,
                                        coherence='c_v')
coherence_model_3train_get = coherence_model_3train.get_coherence()
print(coherence_model_3train_get)


# In[41]:


# calculate coherence metric for test_set (1000 pats dataset)
coherence_model_3test = CoherenceModel(model=model_3, 
                                       texts=processed_docs_1000test, 
                                       dictionary=id_to_word_1000test, 
                                       coherence='c_v')
coherence_model_3test_get = coherence_model_3test.get_coherence()
print(coherence_model_3test_get)


# In[42]:


# calculate coherence metric for each of n topics in test set
coherence_model_3_per_topic = coherence_model_3test.get_coherence_per_topic()
# print(coherence_model_1_per_topic)


# #### Model #3 - Evaluate - Perplexity

# In[43]:


# calculate perplexity metric for model_3 train set
perplexity_model_3train = model_3.log_perplexity(corpus_1000train)
print(perplexity_model_3train)


# In[44]:


# calculate perplexity metric for model_3 test set
perplexity_model_3test = model_3.log_perplexity(corpus_1000test)
print(perplexity_model_3test)


# ### Model #3 - Predict

# #### Model #3 - Predict - Pickle model

# In[45]:


# # update path with location to save pickled model #3
path_pickle_model_3 = '/Users/lee/Documents/techniche/techniche/data/model_3.pkl'

# pickle model #3
pickle.dump(model_3, open(path_pickle_model_3,'wb'))


# In[46]:


model_3 = pickle.load(open(path_pickle_model_3,'rb'))


# #### Model #3 - inference

# In[47]:


predict_input_1_model_3 = get_topics(id_to_word_1000train.doc2bow(text_input_1), 
                                     model_3, 
                                     k=10)
# uncomment below to view predict_input_1_model_3
# predict_input_1_model_3


# In[48]:


predict_input_2_model_3 = get_topics(id_to_word_1000train.doc2bow(text_input_2),
                                     model_3, 
                                     k=10)
# uncomment below to view predict_input_2_model_3
# predict_input_2_model_3


# ## Train Model #4: Gensim LDA model
# Model #4: implementation: Gensim LDAmodel; k_topics=15; n_docs=1000, partition = 80/20
# This model increases the k_topics to 15, relative to model #1 and model #3 above

# In[49]:


# construct model #4
model_4 = LdaModel(corpus=corpus_1000train,
                   id2word=id_to_word_1000train,
                   num_topics=15, 
                   random_state=100,
                   update_every=1,
                   chunksize=100,
                   passes=10,
                   alpha='auto',
                   per_word_topics=True)


# ### Model #4 - Explore and visualize

# In[50]:


# explore topics visually
pyLDAvis.enable_notebook()
viz_topics_model_4 = pyLDAvis.gensim.prepare(model_4, 
                                             corpus_1000train, 
                                             id_to_word_1000train)
# viz_topics_model_1


# In[51]:


# uncomment below to view keywords in n topics in corpus
# pprint(model_4.print_topics())


# In[52]:


# uncomment below to view keywords/weights for topic with idx 0)
# pprint(model_4.print_topic(4))


# ### Model #4 - Evaluate

# #### Model #4 - Evaluate - Coherence

# In[53]:


# calculate coherence metric for train set (1000 pats dataset)
coherence_model_4train = CoherenceModel(model=model_4,
                                        texts=processed_docs_1000train,
                                        dictionary=id_to_word_1000train,
                                        coherence='c_v')
coherence_model_4train_get = coherence_model_4train.get_coherence()
print(coherence_model_4train_get)


# In[ ]:


#TODO (Lee)
# calculate coherence metric for test_set (n = 200 docs/1000 docs total in dataset)
# coherence_model_4test = CoherenceModel(model=model_4,
#                                        texts=processed_docs_1000test,
#                                        dictionary=id_to_word_1000test,
#                                        coherence='c_v')
# coherence_model_4test_get = coherence_model_4test.get_coherence()
# print(coherence_model_4test_get)


# In[ ]:


#TODO (Lee)
# calculate coherence metric for each of the n topics in the test set
# coherence_model_4_per_topic = coherence_model_4test.get_coherence_per_topic()
# print(coherence_model_1_per_topic)


# #### Model #4 - Evaluate - Perplexity

# In[ ]:


# calculate perplexity metric for model_1 train set
perplexity_model_3train = model_3.log_perplexity(corpus_1000train)
print(perplexity_model_3train)


# In[ ]:


# calculate perplexity metric for model_1 test set
perplexity_model_3test = model_3.log_perplexity(corpus_1000test)
print(perplexity_model_3test)


# ### Model #4 - Predict

# #### Model #4 - Predict - Pickle model

# In[ ]:


# pickle model
# # update path with location to save pickled model
path_pickle_model_4 = '/Users/lee/Documents/techniche/techniche/data/model_4.pkl'
pickle.dump(model_4, open(path_pickle_model_4,'wb'))


# In[ ]:


model_4 = pickle.load(open(path_pickle_model_4,'rb'))


# #### Model #4 - inference

# In[ ]:


predict_input_1_model_4 = get_topics(id_to_word_1000train.doc2bow(text_input_1), model_4, k=10)
# uncomment below to view predict_input_1_model_4
# predict_input_1_model_4


# In[ ]:


predict_input_2_model_4 = get_topics(id_to_word_1000train.doc2bow(text_input_2), model_4, k=10)
# uncomment below to view predict_input_2_model_4
# predict_input_2_model_4


# ## Train Model #5: Author-topic model
# Model #4: implementation: Gensim AuthorTopicModel; k_topics=15; n_docs=1000, partition = 80/20
# This model increases the k_topics to 15, relative to model #1 and model #3 above

# In[ ]:


# uncomment to view quick visual index to patent number mapping
# for i in raw_data_1000:
#     print(raw_data_1000.index(i), i['patent_number'])


# In[ ]:


# TODO (Lee) review pat_inv_map workflow
# partitions data_1000 to size of training set (80/20 split so grabs first 800 rows)
data_800 = data_1000[:800]

# create inventor-to-doc mapping from original list of dicts in json api response
pat2inv = pat_inv_map(data_800)


# #### Construct author-topic model

# In[ ]:


# construct author-topic model
model_at = AuthorTopicModel(corpus=corpus_1000train,
                            doc2author=pat2inv,
                            id2word=id_to_word_1000train)


# In[ ]:


# construct vectors for authors
author_vecs = [model_at.get_author_topics(author) for author in model_at.id2author.values()]
author_vecs


# In[ ]:


# retrieve topic distribution for author using use model[name] syntax
# each topic has a probability of being expressed given the particular author, 
# but only the ones above a certain threshold are displayed

model_at['7788103-1']


# In[ ]:


# def show_author(name):
#     print('\n%s' % name)
#     print('Docs:', model.author2doc[name])
#     print('Topics:')
#     pprint([(topic_labels[topic[0]], topic[1]) for topic in model[name]])


# In[ ]:


# build mapping from inventor to patent
inv2pat = gensim.models.atmodel.construct_author2doc(pat2inv)


# ### Model #X - Predict

# In[ ]:


# prediction functions that take input of new text string, and predict topic distribution


# ## Train Model #6: Gensim LDA model
# Model #6: implementation: Gensim LDAmodel; k_topics=5; n_docs=1000, partition = 80/20
# This model decreases the k_topics to 5 for human interpretability in webapp

# In[ ]:


# construct model #6
# TODO (Lee) - resolve deprecation warnings
model_6 = LdaModel(corpus=corpus_1000train,
                   id2word=id_to_word_1000train,
                   num_topics=5, 
                   random_state=100,
                   update_every=1,
                   chunksize=100,
                   passes=10,
                   alpha='auto',
                   per_word_topics=True)


# #### Model #6 - Evaluate - Coherence

# In[ ]:


# calculate coherence metric for train set ((n = 800 docs/1000 docs total in dataset))
coherence_model_6train = CoherenceModel(model=model_6,
                                        texts=processed_docs_1000train,
                                        dictionary=id_to_word_1000train,
                                        coherence='c_v')
coherence_model_6train_get = coherence_model_6train.get_coherence()
print(coherence_model_6train_get)


# #### Model #6 - Evaluate - Perplexity

# In[ ]:


# calculate perplexity metric for model_6 train set
perplexity_model_6train = model_6.log_perplexity(corpus_1000train)
print(perplexity_model_6train)


# In[ ]:


# calculate perplexity metric for model_6 test set
perplexity_model_6test = model_6.log_perplexity(corpus_1000test)
print(perplexity_model_6test)


# #### Model #6 - Predict - Pickle model

# In[ ]:


# pickle model
# # update path with location to save pickled model
path_pickle_model_6 = '/Users/lee/Documents/techniche/techniche/data/model_6.pkl'
pickle.dump(model_6, open(path_pickle_model_6,'wb'))


# In[ ]:


model_6 = pickle.load(open(path_pickle_model_6,'rb'))


# #### Model #6 - inference

# In[ ]:


predict_input_1_model_6 = get_topics(id_to_word_1000train.doc2bow(text_input_1), model_6, k=3)
# uncomment below to view predict_input_1_model_6
predict_input_1_model_6


# In[ ]:


predict_input_2_model_6 = get_topics(id_to_word_1000train.doc2bow(text_input_2), model_6, k=10)
# uncomment below to view predict_input_2_model_6
# predict_input_2_model_6


# In[ ]:




