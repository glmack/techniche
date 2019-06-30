def tokenize_docs(docs):
    """convert words in corpus to word tokens"""
    tokenized_docs = []
    for doc in docs:
        tokenized_docs.append(word_tokenize(doc))
    return tokenized_docs

def clean_docs(tokenized_docs):
    """clean corpus of punctuation"""
    clean_docs = []
    for doc in tokenized_docs:
       clean_docs.append([word for word in doc if word.isalpha()]) 
    return clean_docs

def lower_words(docs):
    """convert words in corpusto lowercase"""
    lowered_words = []
    for doc in docs:
        lowered_words.append([word.lower() for word in doc])
    return lowered_words

def remove_stopwords(clean_docs):
    """remove standard stopwords from corpus"""
    filtered_docs = []
    for doc in clean_docs:
       filtered_docs.append([word for word in doc if word not in stop_words])
    return filtered_docs

def bigrams(docs):
    """create bigrams from corpus"""
    return [bigram_model[doc] for doc in docs]

def trigrams(docs):
    """create trigrams from corpus"""
    return [trigram_model[bigram_model[doc]] for doc in docs]

def lemmatize_docs(docs, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """lemmatize documents"""
    lemmatized_docs = []
    for doc in docs: 
        lemmatized_docs.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return lemmatized_docs

def convert_bytes(num, suffix='B'):
    """ convert bytes int to int in aggregate units"""
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)



