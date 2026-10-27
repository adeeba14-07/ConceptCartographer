"""A short, original sample chapter on Natural Language Processing used for the demo/tests.

Pages are separated by form-feed characters so page numbers appear in the concept frames.
"""

SAMPLE_CHAPTER = """Chapter 4: Language Modelling, Meaning and Topics

Text Preprocessing

Natural Language Processing (NLP) turns raw text into structured information that programs can use. Tokenization is the first step of every NLP pipeline. Tokenization splits a text into tokens such as words, numbers and punctuation marks. Stop word removal deletes very frequent words like "the" and "of" because stop words carry little meaning. Stemming reduces a word to a crude root by chopping suffixes. Lemmatization is more careful than stemming because lemmatization uses a dictionary to return the proper base form of a word. Part-of-Speech (POS) tagging assigns a grammatical category such as noun or verb to every token. POS tagging uses tokenization as its input. A text corpus is a large collection of documents. A text corpus is the raw material for every statistical language model. Text preprocessing improves the quality of every later model. Bag of words ignores word order and counts how often each word occurs. Term Frequency-Inverse Document Frequency (TF-IDF) weights a word by how rare the word is across documents. TF-IDF depends on tokenization and on a text corpus.

\fN-Gram Language Models

A language model assigns a probability to a sequence of words. An N-gram model is a type of language model. An N-gram model uses tokenization to split text into words before counting. An N-gram model estimates the probability of a word from the previous N-1 words. A bigram model conditions each word on one previous word. A trigram model conditions each word on two previous words. The Markov assumption says that a word depends only on a short history. An N-gram model relies on the Markov assumption. Counting N-grams in a large text corpus gives the maximum likelihood estimate of each probability. Data sparsity is the main weakness of an N-gram model because most N-grams never occur in the text corpus. Smoothing fixes data sparsity by moving probability mass to unseen N-grams. Laplace smoothing is a type of smoothing. Laplace smoothing adds one to every count. Backoff smoothing falls back to a shorter N-gram when a longer N-gram is missing. Kneser-Ney smoothing is a type of smoothing that often performs best.

\fLanguage Model Evaluation

Perplexity measures how well a language model predicts a held-out test set. A lower perplexity means a better language model. Perplexity is the inverse probability of the test set normalized by the number of words. Perplexity evaluates an N-gram model on text that the model has never seen. Smoothing lowers the perplexity of an N-gram model on unseen text. Extrinsic evaluation tests a language model inside a downstream application. Intrinsic evaluation uses perplexity because intrinsic evaluation needs no downstream application. Cross entropy is closely related to perplexity. A held-out test set must never be used for training.

\fSyntax and Predicate-Argument Structure

A parser analyses the grammatical structure of a sentence. Dependency parsing is a type of parsing. Dependency parsing links every word to its head word. A dependency parse is the main input for predicate-argument extraction. Predicate-argument structure identifies the action of a sentence and the entities involved in the action. Predicate-argument structure uses dependency parsing to find the subject and the object of a verb. The predicate is the verb that expresses an action or a relation. An argument is an entity that takes part in the predicate. Semantic role labeling assigns a role such as agent or patient to each argument. Semantic role labeling builds on predicate-argument structure. Named entity recognition finds names of people, places and organizations. Named entity recognition uses POS tagging as a feature.

\fMeaning Representation

A meaning representation stores the meaning of a sentence in a machine-readable format. First-order logic is a type of meaning representation. First-order logic expresses a predicate and its arguments as a logical formula. A semantic frame is a type of meaning representation. A semantic frame groups a predicate with its roles. Meaning representation depends on predicate-argument structure. Meaning representation supports question answering and knowledge graph construction. A knowledge graph connects concepts with labelled relations. A knowledge graph is built from triples. A triple contains a subject, a predicate and an object. Triple extraction uses predicate-argument structure to find triples in a sentence.

\fTopic Modelling

Topic modelling discovers the hidden themes in a collection of documents. Latent Dirichlet Allocation (LDA) is a type of topic modelling. LDA treats every document as a mixture of topics. LDA treats every topic as a distribution over words. LDA uses a Dirichlet prior to keep the mixtures sparse. LDA is trained with Gibbs sampling or variational inference. LDA uses bag of words as its input. A topic is a group of words that often occur together. The coherence score measures whether the top words of a topic make sense together. The coherence score evaluates the quality of an LDA model. A higher coherence score means a more interpretable topic. Choosing the number of topics is easier with the coherence score. The c_v coherence score is a type of coherence score. Perplexity also evaluates an LDA model but perplexity correlates poorly with human judgement. LDA groups related concepts into topics. Topic modelling helps readers navigate a large text corpus.
"""
