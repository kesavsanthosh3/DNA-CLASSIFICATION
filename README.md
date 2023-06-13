# DNA-CLASSIFICATION

### Keywords:
### DNA classification, ML models, k-mers, Jaccard similarity, expert system, genomics, dataset, Naive Bayes, Random Forest, SVM, XGBoost, DNA class balance, visualization, hyperparameter tuning, medicine, forensic science, agriculture, performance, existing methods, inference engine, efficiency, effectiveness, impact.

<img src="https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/25a94ca3-fcc2-495f-a24d-993fc8fc2021">

DNA classification refers to the process of categorizing DNA sequences based on their characteristics and features, such as their nucleotide composition, length, and similarity. It is used in various fields, such as genetics, bioinformatics, and forensics. DNA classification is a crucial task in the field of genomics, and this project aims to develop an efficient ML-based DNA classification model. The proposed work aims to develop an efficient ML-based DNA classification model using k-mers, Jaccard similarity, and an expert system for model selection. The dataset used in this project contains 4200 rows of strings with a maximum of 2600 characters of DNA sequences of a human. The project involves performing DNA class balance, visualizing DNA sequences, and hyperparameter tuning for Naive Bayes, Random Forest, SVM, and XGBoost models. The proposed work is essential because DNA classification is a crucial task in the field of genomics, with applications in medicine, forensic science, and agriculture. Moreover, this project's approach can handle large datasets and select the most suitable model based on the given dataset's characteristics using the expert system's inference engine. The results of the proposed model demonstrate better performance in DNA classification tasks compared to existing methods. The expert system accurately recommends the most appropriate ML model for different datasets and features. Compared to existing methods, this project's expert system provides an efficient and effective solution for DNA classification tasks, making it a valuable tool in the field of genomics. Overall, this project provides an efficient and effective solution for DNA classification tasks, which can have a significant impact on various fields like medicine, forensic science, and agriculture.
## INTRODUCTION

The Human Genome Project, completed in 2003, paved the way for the study of genomics, the field of genetics that aims to understand the structure, function, and evolution of genomes. One of the crucial tasks in genomics is DNA classification, which involves categorizing DNA sequences based on their features and characteristics. DNA classification has many applications in fields such as medicine, agriculture, and forensic science, making it a crucial task in genomics.
Machine learning (ML) has been increasingly used in genomics for DNA classification. ML models can efficiently classify large and complex DNA datasets and extract relevant features for classification. In this research paper, we propose an ML-based DNA classification model using k-mers, Jaccard similarity, and an expert system for model selection.The proposed work involves the use of a dataset containing 4200 rows of strings with a maximum of 2600 characters of DNA sequences of a human. To develop an efficient model, the dataset is balanced, and DNA sequences are visualized for deeper understanding. K-mers with DNA sequences with k values as 1, 4, and 6 are used, and Jaccard similarity is used to check for similarity in two DNA sequences. Naive Bayes, Random Forest, SVM, and XGBoost models are implemented and hyperparameter tuning is performed to improve their performance.Moreover, the proposed work incorporates an expert system that gives certain weight to the measures of performance of ML models and calculates an overall score for each model. Based on the inference engine in the expert system, we can select the most appropriate ML model for different datasets and features. The proposed work provides an efficient and effective solution for DNA classification tasks, which can have a significant impact on various fields like medicine, forensic science, and agriculture


## PROJECT ARCHITECTURE
![image](https://user-images.githubusercontent.com/95330456/236203679-6131f41a-1491-4e9e-93d3-e48afe229dc9.png)

## NOVELTY
The proposed project introduces several novel contributions to the field of DNA classification using ML models. Firstly, the project utilizes k-mers and Jaccard similarity to perform DNA classification. K-mers are a powerful method for analyzing DNA sequences, which break down the sequence into smaller subsequences, allowing for better classification accuracy. Jaccard similarity is used to compare the similarity of two DNA sequences and assess their similarity score. The combination of these methods enhances the accuracy and efficiency of DNA classification. Secondly, the proposed project incorporates an expert system to select the most appropriate ML model for the given dataset and features. The expert system weighs the performance of each model and calculates an overall score, which helps in determining the best model for the dataset. This approach provides a more efficient and effective solution for DNA classification tasks compared to existing methods that require manual selection of models. Thirdly, the project visualizes the DNA sequences, providing a deeper understanding of the data and aiding in the selection of appropriate features for classification. This is a crucial contribution as visualizing DNA sequences is a complex task that requires domain expertise. Finally, the project's ability to handle large datasets and select the most suitable model based on the given dataset's characteristics using the expert system's inference engine is a significant advancement in the field. This allows for more efficient and accurate DNA classification, which has far-reaching applications in medicine, forensic science, and agriculture. Overall, the proposed project introduces several novel contributions to DNA classification using ML models, making it a valuable tool in the field of genomics. The combination of k-mers, Jaccard similarity, expert system, and visualization of DNA sequences enhances the accuracy and efficiency of DNA classification tasks, providing a more efficient and effective solution compared to existing methods.

## RESULTS

### Fig 1 : Distribution of lengths of DNA sequences

<img src="https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/42997693-1cc5-46b9-841c-badd63ceaf84" height=400 width=400>

### Fig 2 : Jaccard Similarity Heatmap

<img src="https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/cf193e9c-bdb4-4aac-94bb-e3f395b0ef27" height=400 width=400>

### Fig 3 : Class Balance

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/5de20544-6a54-4466-810c-15d4f41eb5a6 height=400 width=400>

### Fig 4 : Performance Analysis of Naive Bayes Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/5cb91177-0bba-49d4-9a90-32b28ca9c1f3 height=400 width=400>

### Fig 5 : ROC Curve for Naive Bayes Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/446a418f-96f0-4747-810d-d2ae889a653f height=400>

### Fig 6 : Performance Analysis of Random Forest Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/e6c2186e-11a3-4017-9d29-b53b8a92c128 height=400 width=400>

### Fig 7 : ROC Curve for Random Forest Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/0f106443-6e0d-4b74-8623-d7d27079b7ec>

### Fig 8 : Performance Analysis of SVM Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/c853caa3-3e9a-4a98-9e98-d0dc0aac113d
 height=400 width=400>

### Fig 9 : ROC Curve for SVM Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/54fb9633-972b-44f2-a566-109d048e5f29>

### Fig 10 : Performance Analysis of XGBoost Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/5a36b17c-1ae3-4233-93c5-b5356f67072e height=400 width=400>

### Fig 11 : ROC Curve for XGBoost Model

<img src=https://github.com/kesavsanthosh3/DNA-CLASSIFICATION/assets/95330456/c3bfba8e-b961-4294-ad92-adc1b62470ea height=400>




