def tokenize_docs(documents):
    """apply work tokenization to corpus of documents"""
    tokenized_docs = documents.map(word_tokenize)
    return tokenized_docs

