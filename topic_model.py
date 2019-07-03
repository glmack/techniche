
def get_patents_by_month(begin_date,end_date, pats_per_page):
    """ requests patent data from PatentsView API by date range"""
    endpoint_url = 'http://www.patentsview.org/api/patents/query'
    page_counter=1
    data = []
    results = {}
    count=1
    
    for i in range(round(100000/pats_per_page)): # TODO (Lee) - replace with datetime for begin_date to end_date
        
        if count ==0:
            print("error/complete")
            break
            
        elif count > 0:     
            # build query
            query = {"_and":[{"_gte":{"patent_date":"2017-01-01"}},{"_lte":{"patent_date":"2017-01-31"}}]}
            fields=pat_fields
            options={"page": page_counter, "per_page":pats_per_page}
            sort=[{"patent_date":"desc"}]
            params={'q': json.dumps(query),
                    'f': json.dumps(fields),
                    'o': json.dumps(options),
                    's': json.dumps(sort)
                        }
    
            # request and results
            response = requests.get(endpoint_url, params=params)
            status = response.status_code
            print("status:", status,';',"page_counter:",page_counter, ";", "iteration:",i)
            results = response.json()
            count = results.get("count")
            total_pats = results.get("total_patent_count")
            print("patents on current page:",count,';', "total patents:",total_pats)
            data.extend(results)
            page_counter+=1
        
    return data
            # TODO (Lee) results =  json.loads(response.content)
            # TODO (Lee) places.extend(results['results'])
            # TODO (Lee) time.sleep(2)

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



