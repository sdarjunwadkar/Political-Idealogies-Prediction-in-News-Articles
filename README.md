# Political-Idealogies-Prediction-in-News-Articles


## Overview
The project 'Political Ideologies Prediction in News Articles' addresses the impact of diverse media platforms on shaping perspectives while acknowledging the dangers of biased news distorting reality and eroding trust. By leveraging PySpark, NLP, and ML models, the project aims to predict bias in news articles swiftly and adaptably. Integrated with the New York Times API, it showcases the pipeline's functionality by predicting bias in top political articles.

Business goals encompass empowering media literacy by revealing potential political leanings in consumed news, fostering a more informed understanding of media perspectives. Additionally, insights derived could shape regulations, ensuring fairness and accuracy in media reporting, and aid electoral bodies in assessing news impartiality during critical periods.

The prediction goals are twofold: building an effective model for classifying bias (Left, Center, Right) in news articles and implementing real-time bias detection using NYT. The project focuses on content-based analysis, extracting nuanced cues and language patterns indicative of political inclinations to uncover deeper linguistic indicators of bias within articles.


## Dataset Description
The dataset, sourced from the WebIS research group's GitHub repository [2], is a JSON corpus
containing 7,775 news articles with 8 attributes. The "Allside_bias" attribute serves as the target
variable, representing the article's political bias, while the "Content" attribute acts as the predictor,
providing the textual content for analysis. On average, the articles in the corpus span around 4500 words,
yet some articles extend to 16,000 characters. Within this dataset, approximately 1000 articles are from
CNN (Web News), followed by Fox News, and the New York Times contributing 800 articles.


## Setup

Note: Google Colab and Google Drive was used for the project

1) Stable Versions of PySpark & SparkNLP
```
#Install PySpark and SparkNLP specific versions (stable versions)
!pip install spark-nlp==5.1.4 pyspark==3.3.1
```

2) NLTK Toolkit
```
# Install nltk
! pip install nltk
```

3) Library to work with NYT API 
```
# Install the package to interact with NYT API
!pip install pynytimes
```

4) All other packages used
```
# create Spark Session and Spark Context
from pyspark.sql import SparkSession
#Import sparknlp library
import sparknlp

#Import all the required libraries and packages
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassificationModel, RandomForestClassifier
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer
from sparknlp.annotator import Normalizer
import nltk
from nltk.corpus import stopwords
from sparknlp.annotator import StopWordsCleaner
from sparknlp.base import Finisher
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import pynytimes
from pynytimes import NYTAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


## TF-IDF Method Notebook
This notebook contains code for implementation of prediction model to classify a news article as "left", "right" and "center" biased. Following are the contents of this notebook:

1) Setup: Install all packages and dependencies including Spark-NLP and compatible version of pyspark, pynytimes (for NYT API) etc.<br>
2) Import Data : Import the JSON file<br>
3) Initilize All Annotators in Spark-NLP<br>
4) Setup a Preprocessing Pipeline<br>
5) Stratified Train-Test Split<br>
6) Multinomail Naive Bayes (Training Model): Contains Performance Evaluation<br>
7) Random Forest Classifier Model (Training Model): Contains Performance Evaluation<br>
8) Implementation NYT: Demonstration of application using NYT API, Feature importance for both the models and testing with specific examples<br>


### Instructions

1) Please update the path in the import data section to where you have stored the json file<br>
2) Also update the path in the implementation section to where you have stored the 2 trained models<br>
3) If you only want to see the implementation, please do not run the Stratified Train-Test Split, Multinomial Naive Bayes and Random Forest Classifier Section as that will start retraining the model and it will take time.<br>
4) If you are running this notebook in Google Colab Free version and you want to retrain the model, run only one model at a time.<br>


## NYT Data Pull Notebook

Experimenting with the New York Times API by pulling data about, top stories, articles based on some query and most viewed articles.
Using "pynytimes" to work with the API. Please make sure you have registered for the API on [3].
Reference: [4], [5]


## References
1. Beutel, Alexander, et al. "Analyzing Political Bias and Unfairness in Language Models: A Case
Study of GPT-3." Paper with Code, 2023. https://paperswithcode.com/paper/analyzing-political-
bias-and-unfairness-in <br>
2. Weßler, Christoph, et al. "NLPCSS-20 Dataset and Code."
GitHub, 2023. https://github.com/webis-de/NLPCSS-20/tree/main/data <br>
3. The New York Times. "New York Times Developer API." https://developer.nytimes.com/apis <br>
4. den Heijer, Michael. "pynytimes." https://github.com/michadenheijer/pynytimes <br>
5. "Top Stories - pynytimes." https://pynytimes.michadenheijer.com/popular/top-stories <br>
6. Spark NLP. "Spark NLP for Natural Language
Processing." https://sparknlp.org/docs/en/quickstart <br>
7. Apache Spark. "Machine Learning Guide." https://spark.apache.org/docs/latest/ml-guide.html <br>
8. Apache Spark. "Spark Python API." https://spark.apache.org/docs/latest/api/python/index.html <br>
9. Public APIs. "News." Public APIs, 2023. https://github.com/public-apis/public-apis#news
