def get_ml_patents():
    """ builds api query for initial ml dataset"""
    import json
    import requests
    query={"_or":[{"_text_phrase":{"patent_title":"natural language"}},
                  {"_text_phrase":{"patent_abstract":"natural language"}},
                  {"_text_phrase":{"patent_abstract":"machine learning"}},
                  {"_text_phrase":{"patent_title":"machine learning"}},
                  {"_text_phrase":{"patent_abstract":"computer vision"}},
                  {"_text_phrase":{"patent_abstract":"computer vision"}}
    ]}
    # uncomment to use alternate query options
    # query={"cpc_subgroup_id":"G06T3/4046"}
    # query = {"_and":[{"_gte":{"patent_date":"2017-01-01"}},{"_lte":{"patent_date":"2017-01-31"}}]}
    # query={"_and":
    #         [{"_or":
    #             [{"_text_phrase":{"patent_title":"machine learning"}}
    #             ,{"_text_phrase":{"patent_abstract":"machine learning"}}]}
    #         ,{"_and":
    #       [{"patent_year":2016}]}]}
    fields=pat_fields
    options={"per_page":2500}
    sort=[{"patent_date":"desc"}]

    params={'q': json.dumps(query),
            'f': json.dumps(fields),
            'o': json.dumps(options),
            's': json.dumps(sort)}

    # request and results
    response = requests.get(endpoint_url, params=params)
    status = response.status_code
    print("status:", status)
    results = response.json()
    count = results.get("count")
    total_pats = results.get("total_patent_count")
    print("patents on current page:",count,';', "total patents:",total_pats)
    return results

def get_patents_by_month(begin_date,end_date, pats_per_page):
    """ requests patent data from PatentsView API by date range"""
    import json
    import requests
    endpoint_url = 'http://www.patentsview.org/api/patents/query'
    page_counter=1
    data = []
    results = {}
    count=1
    pat_fields= ['assignee_city',
    'assignee_country',
    'assignee_county',
    'assignee_county_fips',
    'assignee_first_name',
    'assignee_first_seen_date',
    'assignee_id',
    'assignee_last_name',
    'assignee_last_seen_date',
    'assignee_lastknown_city',
    'assignee_lastknown_country',
    'assignee_lastknown_latitude',
    'assignee_lastknown_location_id',
    'assignee_lastknown_longitude',
    'assignee_lastknown_state',
    'assignee_latitude',
    'assignee_location_id',
    'assignee_longitude',
    'assignee_organization',
    'assignee_sequence',
    'assignee_state',
    'assignee_state_fips',
    'assignee_total_num_inventors',
    'assignee_total_num_patents',
    'assignee_type',
    'cpc_category',
    'cpc_first_seen_date',
    'cpc_group_id',
    'cpc_group_title',
    'cpc_last_seen_date',
    'cpc_section_id',
    'cpc_sequence',
    'cpc_subgroup_id',
    'cpc_subgroup_title',
    'cpc_subsection_id',
    'cpc_subsection_title',
    'cpc_total_num_assignees',
    'cpc_total_num_inventors',
    'cpc_total_num_patents',
    'detail_desc_length',
    'forprior_country',
    'forprior_date',
    'forprior_docnumber',
    'forprior_kind',
    'forprior_sequence',
    'inventor_city',
    'inventor_country',
    'inventor_county',
    'inventor_county_fips',
    'inventor_first_name',
    'inventor_first_seen_date',
    'inventor_id',
    'inventor_last_name',
    'inventor_last_seen_date',
    'inventor_lastknown_city',
    'inventor_lastknown_country',
    'inventor_lastknown_latitude',
    'inventor_lastknown_location_id',
    'inventor_lastknown_longitude',
    'inventor_lastknown_state',
    'inventor_latitude',
    'inventor_location_id',
    'inventor_longitude',
    'inventor_sequence',
    'inventor_state',
    'inventor_state_fips',
    'inventor_total_num_patents',
    'lawyer_first_name',
    'lawyer_first_seen_date',
    'lawyer_id',
    'lawyer_last_name',
    'lawyer_last_seen_date',
    'lawyer_organization',
    'lawyer_sequence',
    'lawyer_total_num_assignees',
    'lawyer_total_num_inventors',
    'lawyer_total_num_patents',
    'nber_category_id',
    'nber_category_title',
    'nber_first_seen_date',
    'nber_last_seen_date',
    'nber_subcategory_id',
    'nber_subcategory_title',
    'nber_total_num_assignees',
    'nber_total_num_inventors',
    'nber_total_num_patents',
    'patent_abstract',
    'patent_date',
    'patent_firstnamed_assignee_city',
    'patent_firstnamed_assignee_country',
    'patent_firstnamed_assignee_id',
    'patent_firstnamed_assignee_latitude',
    'patent_firstnamed_assignee_location_id',
    'patent_firstnamed_assignee_longitude',
    'patent_firstnamed_assignee_state',
    'patent_firstnamed_inventor_city',
    'patent_firstnamed_inventor_country',
    'patent_firstnamed_inventor_id',
    'patent_firstnamed_inventor_latitude',
    'patent_firstnamed_inventor_location_id',
    'patent_firstnamed_inventor_longitude',
    'patent_firstnamed_inventor_state',
    'patent_kind',
    'patent_number',
    'patent_processing_time',
    'patent_title',
    'patent_type',
    'patent_year',
    'pct_102_date',
    'pct_371_date',
    'pct_date',
    'pct_docnumber',
    'pct_doctype',
    'pct_kind',
    'rawinventor_first_name',
    'rawinventor_last_name',
    'wipo_field_id',
    'wipo_field_title',
    'wipo_sector_title',
    'wipo_sequence']

    query = {"_and":[{"_gte":{"patent_date":begin_date}},{"_lte":{end_date:"2019-01-01"}}]}
    
    fields=pat_fields
    options={"page": page_counter, "per_page":pats_per_page}
    sort=[{"patent_date":"desc"}]
    params={'q': json.dumps(query),
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
    # print("status:", status,';',"page_counter:",page_counter,) # ";", "iteration:",i
    results = response.json()
    # extract data from response
    data_response = results['patents']
    count = results.get("count")
    total_pats = results.get("total_patent_count")
    # print("patents on current page:",count,';', "total patents:",total_pats)
    data.append(data_response)

    while (total_pats/pats_per_page*page_counter): # TODO (Lee) - replace with datetime for begin_date to end_date
        if count ==0:
            print("error/complete")
            break
            
        elif count > 0:     
            # build query
            query = {"_and":[{"_gte":{"patent_date":begin_date}},{"_lte":{end_date:"2019-01-01"}}]}

#             {"_or":[{"cpc_subgroup_id": "G06T3/4046"},
#                     {"cpc_subgroup_id": "G06T9/002"}]}
            fields=pat_fields
            options={"page": page_counter, "per_page":pats_per_page}
            sort=[{"patent_date":"desc"}]
            params={'q': json.dumps(query),
                    'f': json.dumps(fields),
                    'o': json.dumps(options),
                    's': json.dumps(sort)
                        }
    
            # request and results
            # response = requests.get(endpoint_url, params=params)
            # status = response.status_code
            # print("status:", status,';',"page_counter:",page_counter,) # ";", "iteration:",i
            # results = response.json()
            # extract data from response
            # data_response = results['patents']
            count = results.get("count")
            # total_pats = results.get("total_patent_count")
            # print("patents on current page:",count,';', "total patents:",total_pats)
            data.extend(data_response)
            page_counter+=1
        
        else:
            print("error #2/complete")
            break
        
    return data
            # TODO (Lee) ? results =  json.loads(response.content)
            # TODO (Lee) ? places.extend(results['results'])
            # TODO (Lee) ? time.sleep(2)

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



