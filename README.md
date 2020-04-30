# case_law_g45
Final Project for CS109B (Data Science for Case Law) - Group 45

 - **Ignite talk slides uploaded by May 9 11:59 EDT (midnight)**
 - **Final project due May 10 11:59 EDT (midnight)**

## Group Members

| Name          |  |   |
|:--------------|:-|:-:|
| Gufran Pathan |  |   |
| Prerna Aggarwal |  |   |
| Fernando Medeiros |  |   |

## Links:

[Module C: Data Science for Caselaw Project Data Link](https://drive.google.com/drive/folders/1Dvtk_rxNK-4tXYmRWZhX2no9TrFTu8SD)

[10 Ways Your Data Project Can Fail](https://drive.google.com/file/d/1I9ut6aRU9L9UNy83uA03rblIY7pl7GT1/view)

[NLP Progress Summarization](http://nlpprogress.com/english/summarization.html)

**Suggested extractive reading list from the NLP Progress Summarization link:**

 1. [Learning to Extract Coherent Summary via Deep Reinforcement Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16838/16118)
 1. [Extractive Summarization with SWAP-NET: Sentences and Words from Alternating Pointer Networks](http://aclweb.org/anthology/P18-1014) (Fernando)
  >>"A fundamental requirement in any extractive summarization model is to identify the salient sentences that represent the key information mentioned...Despite their popularity, neural networks still have some issues while applied to document summarization task. These methods lack the latent topic structure of contents. Hence the summary lies only on vector space that can hardly capture multi-topical content. Another issue is that the most common architectures for Neural Text Summarization are variants of recurrent neural networks (RNN) such as Gated recurrent unit (GRU) and Long short-term memory (LSTM). These models have, in theory, the power to ‘remember’ past decisions inside a fixed-size state vector; however, in practice, this is often not the case."  
  >>"The contribution of this paper is proposing a general neural network-based approach for summarization that extracts sentences from a document by treating the summarization problem as a classification task. The model computes the score for each sentence towards its summary membership by modeling abstract features such as content richness, salience with respect to the document, redundancy with respect to the summary and the positional representation. The proposed model has two improvements that enhance the summarization effectiveness and accuracy: (i) it has a hierarchical structure that reflects the hierarchical structure of documents; (ii) while building the document representation, two levels of self-attention mechanism are applied at word-level and sentence-level. This enables the model to attend differentially to more and less important content."  
 1. [A Hierarchical Structured Self-Attentive Model for Extractive Document Summarization (HSSAS)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344797)
 1. [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230) (Gufran)
 SumaRunner uses an RNN for summarization. It has two architectures, one for extractive summarization and another for abstractive summarization. Since most labelled datasets for summarization are 'abstractive' in nature (i.e. paraphrased summaries have been generated by humans), the paper presents a method to generate extractive labels from abstractive summaries. They do this by adding sentences from the main text to the summaries greedily such that the ROGUE score increases. When there is no further increase in the rogue score, the collected set of sentences is returned with a label of '1' and others as '0'. The algorithm operates at two levels - first at the word level and second at the sentence level. Word embeddings are first generated using a bi-directional rnn. Embeddings are developed for each word and then fed in to the next layer ('sentence layer'), which generates sentence embeddings. The output is a binary 0/1 for each sentence indicating whether the sentence should be included in the summary or not. In the abstractive version, the output is replaced with an RNN decoder to generate the summary. Another contribution of the paper is that the weights makes the model interpratable - they provide scores for why the sentence is chosen based on different criteria - novelty, salience etc.
 1. [Extractive Summarization as Text Matching](https://arxiv.org/abs/2004.08795) (Fernando)
  >>"In this paper, we propose a novel summary-level framework (MATCHSUM, Figure 1) and conceptualize extractive summarization as a semantic text matching problem. The principle idea is that a good summary should be more semantically similar as a whole to the source document than the unqualified summaries...Instead of following the commonly used framework of extracting sentences individually and modeling the relationship between sentences, we formulate the extractive summarization task as a semantic text matching problem, in which a source document and candidate summaries will be (extracted from the original text) matched in a semantic space."  
  >>"Specific to extractive summarization, we propose a Siamese-BERT architecture to compute the similarity between the source document and the candidate summary. Siamese BERT leverages the pre-trained BERT (Devlin et al., 2019) in a Siamese network structure (Bromley et al., 1994; Hoffer and Ailon, 2015; Reimers and Gurevych, 2019) to derive semantically meaningful text embeddings that can be compared using cosine-similarity. A good summary has the highest similarity among a set of candidate summaries."  
 1. [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) (Gufran)

There are 4 states with available data and can be downloaded from this link: [Caselaw Access Project/Bulk Data](https://case.law/bulk/download/)

### Additional Links

 -[Comprehensive Guide to Text Summarization using Deep Learning in Python](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)
 -[Deep learning for specific information extraction from unstructured texts](https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada)
 -[Extractive Text Summarization Using Neural Networks](https://heartbeat.fritz.ai/extractive-text-summarization-using-neural-networks-5845804c7701)

## Techniques

We should pick up three to four techniques to try.  
 Do we have enough time to try several?  
  Yeah I think 3 and 4 are difficult and unlikely to yield good results. We could do it as a trial and say we wanted to test out how good a first attempt is.  

 1. Extractive summarization using classification
 1. Unsupervised learning using Page rank
 3. Seq2Seq
 4. Transofrmers

## Task List:

 - [x] Create task list (Fernando)
 - [ ] Clean and prepare data (names)
  - Remove HTML tags
  - Remove headnotes from the training dataset
 - [ ] Tokenize and pad the data (name)
 - [ ] Develop model (names)
 - [ ] Train/tune model (names)
 - [ ] Model diagnostics (names)

## Python Libraries

 - lzma
 - lxml

## Data Catalog:

 - Casebody in XML
 - Casebody includes headnotes

## Open Questions:

 - Where are headnotes in the dataset? The new North Carolina data includes the headnotes--redownload the data from the project link above.
 - Why are there more headnotes than cases? _Headnotes are summaries of legal principles which each case have many. Lawyers use headnotes as shortcuts for looking for legal principles to use and what other cases have similar headnotes_
 - **It appears that the headnotes are different from the casebody, excluding the headnotes. Can we use the extractive approach since the headnotes are more abstractive than extractive?**
 - **Do we use other dataset fields other than the casebody?**
 - **What format should the casebody take? Should it contain all the text, minus the XML tags and headnotes, in one long string?**
