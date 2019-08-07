
def get_patent_fields_list():
    """scrape patent fields that are retrievable from PatentsView API"""
    import requests
    from bs4 import Tag, NavigableString, BeautifulSoup
    import pandas as pd
    url = "http://www.patentsview.org/api/patent.html"
    page = requests.get(url)
    table = []
    table_fieldnames = []
    soup = BeautifulSoup(page.text, 'lxml')
    table = soup.find(class_='table table-striped documentation-fieldlist')
    table_rows = table.find_all('tr')
    counter = 0
    for tr in table_rows:
        if counter == 0:
            th = tr.find_all('th')
            row = [tr.text for tr in th]
            table.append(row)
            counter += 1

        elif counter > 0:
            td = tr.find_all('td')
            row = [tr.text for tr in td]
            table.append(row)

    for row in l[1:]:
        table_fieldnames.append(row[0])
    return table_fieldnames


def get_patent_fields_df():
    """return df from scraped patent fields that are
     retrievable from PatentsView API"""
    import requests
    from bs4 import Tag, NavigableString, BeautifulSoup
    import pandas as pd
    url = "http://www.patentsview.org/api/patent.html"
    page = requests.get(url)
    table = []

    soup = BeautifulSoup(page.text, 'lxml')
    table = soup.find(class_='table table-striped documentation-fieldlist')
    table_rows = table.find_all('tr')
    counter = 0
    for tr in table_rows:
        if counter == 0:
            th = tr.find_all('th')
            row = [tr.text for tr in th]
            table.append(row)
            counter += 1

        elif counter > 0:
            td = tr.find_all('td')
            row = [tr.text for tr in td]
            table.append(row)

    df = pd.DataFrame(table)
    return df


def get_ml_patents():
    """builds api query for initial ml dataset"""
    import json
    import requests
    endpoint_url = 'http://www.patentsview.org/api/patents/query'
    query = {"_or": [{"_text_phrase": {"patent_title": "natural language"}},
                     {"_text_phrase": {"patent_abstract": "natural language"}},
                     {"_text_phrase": {"patent_abstract": "machine learning"}},
                     {"_text_phrase": {"patent_title": "machine learning"}},
                     {"_text_phrase": {"patent_abstract": "computer vision"}},
                     {"_text_phrase": {"patent_abstract": "computer vision"}}
                     ]
             }
    # uncomment to use alternate query options
    # query={"cpc_subgroup_id":"G06T3/4046"}
    # query = {"_and":[{"_gte":
    #                   {"patent_date":"2017-01-01"}},
    #                   {"_lte":{"patent_date":"2017-01-31"}}]}
    # query={"_and":
    #         [{"_or":
    #             [{"_text_phrase":{"patent_title":"machine learning"}}
    #             ,{"_text_phrase":{"patent_abstract":"machine learning"}}]}
    #         ,{"_and":
    #       [{"patent_year":2016}]}]}
    pat_fields = get_patent_fields_list()
    fields = pat_fields
    options = {"per_page": 50}
    sort = [{"patent_date": "desc"}]

    params = {'q': json.dumps(query),
              'f': json.dumps(fields),
              'o': json.dumps(options),
              's': json.dumps(sort)}

    # request and results
    response = requests.get(endpoint_url, params=params)
    status = response.status_code
    results = response.json()
    print(results)
    count = results.get("count")
    data_resp = results['patents']
    total_pats = results.get("total_patent_count")
    return data_resp


def get_patents_by_month(begin_date, end_date, pats_per_page):
    """requests patent data from PatentsView API by date range"""
    import json
    import requests
    endpoint_url = 'http://www.patentsview.org/api/patents/query'
    page_counter = 1
    data = []
    results = {}
    count = 1
    pat_fields = get_patent_fields_list()

    query = {"_and": [{"_gte": {"patent_date": begin_date}},
             {"_lte": {end_date: "2019-01-01"}}]}

    fields = pat_fields
    options = {"page": page_counter, "per_page": pats_per_page}
    sort = [{"patent_date": "desc"}]
    params = {'q': json.dumps(query),
              'f': json.dumps(fields),
              'o': json.dumps(options),
              's': json.dumps(sort)
              }

    # response = requests.get(endpoint_url, params = params)
    # results =  json.loads(response.content)
    # places.extend(results['results'])
    # time.sleep(2)
    # request and results
    response = requests.get(endpoint_url, params=params)
    status = response.status_code
    # print("status:",
    #        status,
    #        ';',
    #        "page_counter:",
    #        page_counter,)
    #        ";", "iteration:",i
    results = response.json()
    # extract data from response
    data_response = results['patents']
    count = results.get("count")
    total_pats = results.get("total_patent_count")
    # print("patents on current page:",count,';', "total patents:",total_pats)
    data.append(data_response)

    # TODO (Lee) - replace with datetime for begin_date to end_date
    while (total_pats/pats_per_page*page_counter):
        if count == 0:
            print("error/complete")
            break

        elif count > 0:
            # build query
            query = {"_and": [{"_gte": {"patent_date": begin_date}},
                              {"_lte": {end_date: "2019-01-01"}}]}

#             {"_or":[{"cpc_subgroup_id": "G06T3/4046"},
#                     {"cpc_subgroup_id": "G06T9/002"}]}
            fields = pat_fields
            options = {"page": page_counter, "per_page": pats_per_page}
            sort = [{"patent_date": "desc"}]
            params = {'q': json.dumps(query),
                      'f': json.dumps(fields),
                      'o': json.dumps(options),
                      's': json.dumps(sort)
                      }

            # request and results
            # response = requests.get(endpoint_url, params=params)
            # status = response.status_code
            # print("status:", status,
            #       ';',
            #       "page_counter:",
            #       page_counter,)
            # ";", "iteration:",i
            # results = response.json()
            # extract data from response
            # data_response = results['patents']
            count = results.get("count")
            # total_pats = results.get("total_patent_count")
            # print("patents on current page:",
            #       count,
            #       ';',
            #       "total patents:",
            #       total_pats)
            data.extend(data_response)
            page_counter += 1

        else:
            print("error #2/complete")
            break

    return data
    # TODO (Lee) ? results =  json.loads(response.content)
    # TODO (Lee) ? places.extend(results['results'])
    # TODO (Lee) ? time.sleep(2)


def trim_data(data, keys):
    """subset fields returned from api response"""
    new_data = []
    for dictionary in data:
        new_data.append(
            dict((k, dictionary[k]) for k in keys if k in dictionary))
    return new_data


def create_title_abstract_col(data):
    """creates new col from title and abstract cols of api response"""
    for dictionary in data:
        dictionary['patent_title_abstract'] =
        str([dictionary['patent_title'] +
            '. ' + dictionary['patent_abstract']][0])
    return data


def structure_dataframe(data):
    """ creates dataframe and organizes columns from dictionary"""
    import pandas as pd
    df = pd.DataFrame(data)
    df = df[['patent_number', 'patent_date', 'patent_title_abstract']]
    df.sort_values(by=['patent_date'], ascending=True, inplace=True)
    return df


def partition_dataframe(dataframe, train_pct):
    len(dataframe)
    text_train = dataframe[:round(len(dataframe)*train_pct)]
    text_test = dataframe[round(len(dataframe)*train_pct):]
    return text_train, text_test


def build_pipeline():
    import spacy
    nlp = spacy.load('en_core_web_sm')
    # nlp.add_pipe(token_stats, name="token_stats", first=True)
    # nlp.add_pipe(tokenize, name="tokenize", first=True)
    return nlp


def token_stats(doc):
    print("After tokenization, this corpus has {} tokens.".format(len(doc)))
    print("The part-of-speech tags are:", [token.pos_ for token in doc])
    return doc


def process_docs(text_data):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    from nltk.corpus import stopwords
    """pre-processes patent documents in pipeline"""
    nlp = build_pipeline()
    processed_docs = []
    stop_words = stopwords.words(
        '/Users/lee/Documents/techniche/techniche/data/stopwords/english')
    for doc in nlp.pipe(text_data, batch_size=100):

        # ents = doc.ents  # Named entities.

        # Keep only words (no numbers, no punctuation).
        # Lemmatize tokens, remove punctuation and remove stopwords.
        doc = [token.text for token in doc if token.text.isalpha()]
        # [token.text for token in doc]
        doc = [token.lower() for token in doc]
        # filter stopwords from nltk list of common stopwords
        doc = [token for token in doc if token not in stop_words]

        # Add named entities, but only if compound of more than word.
        # doc.extend([str(entity) for entity in ents if len(entity) > 1])

        processed_docs.append(doc)

    return processed_docs


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
        lemmatized_docs.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return lemmatized_docs


def convert_bytes(num, suffix='B'):
    """ convert bytes int to int in aggregate units"""
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def pat_inv_map(data):
    """builds patent(idx) mapping to list of inventors for a-t model"""
    pat_inv_dict = {}
    for patent in data:
        idx = data.index(patent)
        inventors = [inventor['inventor_id']
                     for inventor in patent['inventors']]
        pat_number = int(patent['patent_number'])
        pat_inv_dict[idx] = inventors
    return pat_inv_dict


def get_topics(doc, k=5, model_lda=model_lda):
    topic_id = sorted(model_lda[doc][0], key=lambda x: -x[1])
    top_k_topics = [x[0] for x in topic_id[:k]]
    return [(i, model_lda.print_topic(i)) for i in top_k_topics]\n