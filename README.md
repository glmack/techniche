# techniche - yesterday’s patent is tomorrow’s business market. 
Techniche offers machine learning-based decision support tools to help business users surface and evaluate market trends via statistical learning of the technical details of machine learning inventions that are described in the text and associated metadata of patent documents.

### Business understanding
Business decision-makers across finance, product and people teams require data on future market trends to reduce uncertainty and take actions to secure early-mover advantages. Techniche addresses this need by providing an additional data-driven signal to future market trends based on the patent pipeline of underlying machine learning technologies.

### Data understanding
Techniche learns from public patent documents of the United States Patent Organization (USPTO) that are made available through the PatentsView API at http://www.patentsview.org/api/doc.html. Interested users can explore this patent data via the graphical user interface at http://www.patentsview.org/query/. Techniche broadens access to analytical insights from patents, a data source that is more commonly limited to researchers or paying business customers of proprietary patent/intellectual property (IP) analytics databases.

### Data preparation
Preprocessing techniques such as word tokenization and punctuation cleaning are applied to raw text data prior to introduction to models.

### Modeling
Techniche predicts technological specializations as a supervised multi-class classification task using neural networks (CNN, RNN), as well as an unsupervised learning of latent technological topics using topical models (LDA). Techniche predicts technological significance as a supervised regression scoring task using neural network (CNN, RNN) models. Techniche leverages predictive power by combining pre-trained word-embeddings trained on generalized text with domain-specific training on patent texts.

### Evaluation
The project evaluates model performance through multiple metrics, including log loss for classification tasks.

### Deployment
Techniche is available for user experimentation on www.techniche.app via a keyword-based Flask web application that offers a search interface where users can input search terms and return predicted “tech niche” topics and their associated entities (e.g. companies, locations), from Techniche topic models. Techniche targets business consumers in the machine learning space who, as users, want to be able to evaluate the probability and significance of possible future market trends in the machine learning space so that they can inform management decisions in the context of uncertainty. Potential users and their associated tasks might include: market analysts conducting competitive analysis, tech investors valuing companies on the basis of IP assets, attorneys preparing IP cases, and policymakers supporting local business clusters.
