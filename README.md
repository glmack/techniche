# techniche

Techniche is a suite of decision support tools to help business users evaluate the probability and significance of future market trends by using patent documents to predict granular technological specializations - tech niches - in machine learning markets. Techniche augments human decision-making by surfacing and evaluating data-driven market signals via statistical learning of the technical details of machine learning inventions described in the text of patent documents.

### Business value
Early-mover advantages depend on the ability of business decision-makers to acquire and triangulate knowledge of the pipeline of machine learning technologies that might shape markets into the future.

### Data sources
Techniche learns from public patent documents of the United States Patent Organization (USPTO). Data is acquired from the USPTO PatentsView API and bulk downloads, and subsequently stored in an Amazon Web Services (AWS) cloud bucket and PostgreSQL database. While USPTO provides a graphical interface for retrieval and exploratory visualization of the patent data contained in the API, Techniche shares analytical insights of the full patent documents that are otherwise commonly limited to specialized research communities or paying customers of proprietary patent analytics databases. 

### Data preparation
Preprocessing techniques - tokenization, punctuation cleaning, and stemming and lemmatization - are applied to raw text data prior to introduction to models.

### Modeling
Techniche predicts technological specializations as a supervised multi-class classification task using neural networks (CNN, RNN), as well as an unsupervised learning of latent technological topics using topical models (LDA). Techniche predicts technological significance as a supervised regression scoring task using neural network (CNN, RNN) models. Techniche leverages predictive power by combining pre-trained word-embeddings trained on generalized text with domain-specific training on patent texts.

### Evaluation
The project evaluates model performance through multiple metrics, including log loss for classification tasks.

### Deployment
Techniche is available for user experimentation on www.predictpatent.com via a keyword-based Flask web application that offers a search interface where users can input search terms and return predicted “tech niche” topics and their associated entities (e.g. companies, locations), from Techniche topic models. Techniche targets business consumers in the machine learning space who, as users, want to be able to evaluate the probability and significance of possible future market trends in the machine learning space so that they can inform management decisions in the context of uncertainty. Potential users and their associated tasks might include: market analysts conducting competitive analysis, tech investors valuing companies on the basis of IP assets, attorneys preparing IP cases, and policymakers supporting local business clusters.
