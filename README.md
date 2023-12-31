# Automated-Evaluator for Descriptive Answers
Automated-Evaluator for Descriptive Answers (AEDA) is a tool for evaluating and scoring of essays written in response to specific prompts. It can be defined as the process of scoring written essays using computer programs. The process of automating the assessment process could be useful for both educators and learners since it encourages the iterative improvements of students' writings. 

## Why AEDA?
Automated grading if proven effective will not only reduce the time for assessment but comparing it with human scores will also make the score realistic. The project aims to develop an automated essay assessment system by use of machine learning techniques and Neural networks by classifying a corpus of textual entities into a small number of discrete categories, corresponding to possible grades.

## Dataset

The dataset we are using is ‘The Hewlett Foundation: Automated Essay Scoring Dataset’ by ASAP. You can find in the below link or download from the Dataset folder. 
 
https://www.kaggle.com/c/asap-aes/data 


## Architecture Diagram

![Model ADA](https://github.com/aditishaktawat/Automated-Evaluator/assets/146921675/f640ff7d-ccc1-4372-aaba-4cb934f571d8)

 
## Proposed Model

The proposed system consists of data collection and annotation, preprocessing module, similarity measurement module, model training module, results predicting module, machine learning model module, and final result predicting module. 
The model achieved is divided into 4 modules as follows:

**1. Data Preprocessing**

We began by doing some standard preprocessing steps like filling in null values and selecting valid features from the entire dataset after a thorough study.Next we plotted a graph to get a measure of the skewness of our data  and applied normalisation techniques to reduce this skewness.The next step involved cleaning the essays to make our training process easier for getting a better accuracy.To achieve this we removed all the  unnecessary symbols ,stop words and punctuations from our essays. To increase our accuracy even more we even planned to add some extra features like the number of sentences , number of words,number of characters, average word length etc. Moreover , we even worked on techniques like getting the noun ,verb ,adjective and adverb counts using parts of speech tagging as well as getting the total misspellings in an essay by comparison with a corpus.We applied various machine learning algorithms on this data as explained in the next section.

Processed dataset can be found in the file, **Processed_data.csv**



**2. Word Vectorization**

For making our data ready to apply algorithms,we require one more step.Machine learning algorithms can not be applied on sentences or words,they can only be used upon numeric data.Our dataset has a field which has essays that need to be converted into a numeric form first in order to train it.
 

 
 **3. Applying Similarity Measures**
 
 **Cosine Similarity**
 Cosine Similarity is a method of calculating the similarity of two vectors by taking the 
 dot product and dividing it by the magnitudes of each vector.Score Prediction Using Cosine Similarity Before Model Suggestions gave an average error of 11.1%.

 **Word Mover's Distance**
 Word Mover’s Distance tries to measure the semantic distance of two documents, and word2vec embeddings bring the semantic measurement.

WMD uses the word embeddings of the words in two texts to measure the minimum distance that the words in one document need to “travel” in semantic space to reach the words in the other document.It gave an error of about 10%.


**4. Machine Learning**

To increase our accuracy even more we add some extra features. The main features selected are based on statistical, semantic and syntactic analysis. These features were tested on different supervised prediction models to find out which model works the best. This is known as Feature Extraction.


Initially we applied machine learning algorithms like linear regression, SVR and Random Forest on the dataset without addition of features that were mentioned in the preprocessing section before. Our results were not really satisfactory as our mean squared error was quite high for all the above algorithms. After this initial evaluation, we added the extra features,applied Word2vec again on this modified dataset and applied the same three algorithms.There was a great improvement in the performance of all three algorithms especially Random forest for which the mean squared error reduced drastically. 

![Features ADA](https://github.com/aditishaktawat/Automated-Evaluator/assets/146921675/54ab14e4-9b3f-4b57-aa0b-0c7b13a10390)

Python notebook for the implementation of this module can be found in the file, **main.ipynb**
 
 ## Conclusion

In this project, we demonstrated an automatic evaluation scheme for subjective answers with performance comparable to the human evaluation. It is based on machine learning and natural language processing techniques. Various similarity and dissimilarity methods are studied, and various other measures such as the keyword’s presence and percentage mapping of sentences are utilized to overcome the abnormal cases of semantically loose answers. The experimentation results show that, on average word2vec approach performs better than traditional word embedding techniques as it keeps the semantics intact. Furthermore, Word Mover’s Distance performs better than Cosine Similarity. In most cases and helps to train the machine learning model faster. With enough training, the model can stand on its own and predict scores without the need for any semantics checking. 

