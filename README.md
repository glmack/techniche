# techniche

### Machine learning-based patent signals for business decisions 
Techniche offers machine learning-based decision support tools to help business users surface technology ideas from patent documents for machine-learning inventions.

### Business understanding
Business decision-makers across finance, product and people teams require data on future market trends to reduce uncertainty and validate strategies to secure early-mover advantages. Techniche addresses this need by providing an additional data-driven signal to future market trends based on the patent pipeline of underlying machine learning technologies.

### Data
#### Data understanding
Techniche broadens access to analytical insights from patents, a data source that is more commonly limited to researchers or paying business customers of proprietary patent/intellectual property (IP) analytics databases. Techniche learns from public patent documents of the United States Patent Organization (USPTO) that are made available through the PatentsView API at http://www.patentsview.org/api/doc.html. Interested users can explore this patent data via the graphical user interface at http://www.patentsview.org/query/.

#### Data preparation
Natural language pre-processing techniques such as word tokenization and punctuation cleaning are applied to raw text data from patent titles and summary descriptions prior to introduction to models.

### Modeling
Techniche models the technological specializations - tech niches - that might become tomorrow's business market. Intially, techniche uses Latent Dirichlet Allocation (LDA), an unsupervised set of topic models to generate the probable range of topics expressed in patent documents for machine learning-based inventions. This data-driven approach to topic discovery is an alternative to technological heuristics based on existing taxonomic systems of patent governance organizations such as USPTO that have recognized limits in capturing continuous changes within and across invention spaces.

#### Evaluation
The project evaluates model performance through perplexity and topic coherence methods, which are the standard approaches to evaluation of unsupervised topic models.

### Deployment
Techniche is available for user experimentation on www.techniche.app via a keyword-based Flask web application that offers a search interface where users can input text strings describning technical areas, and return predicted “tech niche” topics and their associated entities (e.g. companies, locations).
