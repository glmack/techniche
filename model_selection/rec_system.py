def alphanum_to_int(string):
    """encodes alphanumeric string to int"""
    import math
    return int.from_bytes(string.encode(), 'little')

def int_to_alphanum(string):
    """decodes int to alphaumeric string"""
    import math
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

def get_pat_recs(patent_number, cosine_sim=cosine_sim):
    """take user input of string and output most similar documents"""
    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # get idx of patent_number that matches text
    idx = indices[patent_number]

    # calculate pairwise similarity scores of all patents with given patent
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort patents based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # get scores of 10 most similar patents
    sim_scores = sim_scores[1:11]

    # get patent indices
    patent_idxs = [i[0] for i in sim_scores]

    # Return top 10 most similar documents
    return df_1000_2['patent_number'].iloc[patent_idxs]