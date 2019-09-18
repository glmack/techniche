**[summary](#summary) | [contents](#contents) | [usage](#usage) | [running the notebooks](#running-the-notebooks) | [issues](#issues) | [license](#license)**

# techniche

#### Machine learning-based patent signals for technology decisions 
Techniche offers machine learning-based decision support tools to help business users surface technology ideas from patent documents for machine learning inventions.

#### Business understanding
Technology decision-makers - in R&D, people and product - require data to make choices in markets shaped by machine-learning technologies. Techniche recommends technology ideas based on the pipeline of underlying machine learning technologies in patents.

#### Data understanding
Techniche learns from public patent documents of the United States Patent Organization (USPTO) that are made available through the PatentsView API at http://www.patentsview.org/api/doc.html. Interested users can explore and download this patent data via the graphical user interface at http://www.patentsview.org/query/.

##### Data preparation
Natural language pre-processing techniques such as word tokenization and punctuation cleaning are applied to raw text data from patent titles and summary descriptions prior to introduction to models.

Explore notebooks detailing data preparation and modeling in topic_model.ipynb and rec_system.ipynb of the model selection directory.

#### Modeling
Techniche will make recommendations using a hybrid recommender system. At the current stage of development, a collaborative filtering recommender component uses matrix factorization based on the Spark implementation of alternating least squares (ALS). A content-based recommender component, currently under development, addresses the cold start problem associated with making predictions for new users and items. The recommender will use text-based document (item) similarity metrics and also elicit user preferences through the web app. Latent Dirichlet Allocation (LDA), an unsupervised set of topic models is explored to generate the probable range of topics expressed in patent documents for machine learning-based inventions.

#### Evaluation
Recommendations are evaluated in terms of relevance to technology decision-makers. Intermediate intrinsic evaluation metrics, such as coherence and perplexity metrics for LDA, provide additional diagnostics.

#### Deployment
Techniche is available for user experimentation on https://www.techniche.app as a Flask web application that offers a search interface where users can input text strings describing technical areas and return predicted topics and their associated word co-occurences.
