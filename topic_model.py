def tokenize_docs(docs):
    """apply word tokenization to corpus"""
    tokenized_docs = []
    for doc in docs:
        tokenized_docs.append(word_tokenize(doc))
    return tokenized_docs

def clean_docs(tokenized_docs):
    clean_docs = []
    for doc in tokenized_docs:
       clean_docs.append([word for word in doc if word.isalpha()]) 
    return clean_docs

