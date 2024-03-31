# icd9-coding

Automated ICD-9 coding using machine learning (ML) involves the application of ML algorithms to automatically assign International Classification of Diseases, Ninth Revision (ICD-9) codes to medical records or clinical notes. The goal of automated ICD-9 coding is to streamline the process of assigning ICD-9 codes to medical records. Traditionally, this task has been performed manually by medical coders, which can be time-consuming and prone to errors.

We look at the paper "Automated ICD-9 Coding via A Deep Learning Approach" by Min Li et al. which proposes a novel method for ICD-9 coding called “DeepLabeler”. DeepLabeler is a two step method, the first step is feature extraction and the second step is multilabel classification to assign ICD-9 code. The innovation of Deep Labeler lies in the feature extraction step which uses both Word2Vec and Doc2Vec in parallel.

Traditional ML algorithms use Bag of Words (BOW) techniques to vectorize documents for feature extraction. BOW leaves behind semantic information embedded in word order and isn’t suitable for problems like ICD-9 coding where there is a high variance in document length. DeepLabeler solves these issues by using Doc2Vec which captures the semantic information of the whole document and does not leave any words behind. At the time of publishing, the authors suggest that this is the first study to use Doc2Vec for the ICD-9 coding problem.

The second step is multilabel classification. DeepLabeler concatenates the features from the parallel Word2Vec and Doc2Vec pathways and feeds it into a CNN with a fully connected layer followed by a sigmoid activation for the final ICD-9 code assignments.
