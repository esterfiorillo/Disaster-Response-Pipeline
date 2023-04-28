from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle
import sys

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data (database_file):

	"""
	Parameters:
	database_file: file to sqlite database

	Returns:
	X - messages (input variable) 
	y - categories of the messages (output variable)
	"""

	engine = create_engine('sqlite:///' + database_file)
	df = pd.read_sql_table('ELT_Preparation', engine)
	X = df['message']
	y = df[df.columns[4:]]
	
	return X, y 


def tokenize(text):

	"""
	Parameters:
	text - raw messages

	Returns:
	clean_tokens - tokenized messages
	"""

	detected_urls = re.findall(url_regex, text)
	for url in detected_urls:
		text = text.replace(url, "urlplaceholder")

	tokens = word_tokenize(text)
	lemmatizer = WordNetLemmatizer()

	clean_tokens = []
	for tok in tokens:
		clean_tok = lemmatizer.lemmatize(tok).lower().strip()
		clean_tokens.append(clean_tok)

	return clean_tokens


def train_model (X_train, y_train):

	"""
	Parameters:
	X_train - Messages for training
	y_train - Labels with messages categories for training

	Returns:
	cv - pipeline model trained with grid search
	"""

	pipeline = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(AdaBoostClassifier()))
	])

	parameters = {
		'clf__estimator__learning_rate': [0.5, 1.0],
		'clf__estimator__n_estimators': [10, 20]
	}
	cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
	cv.fit(X_train, y_train)
	return cv

def eval_model (X_test, y_test, model, columns):

	"""
	Prints the metrics obteined in the trained model

	Parameters:
	X_test - Messages for testing the model
	y_test - Labels with messages categories for testing the model
	model - model trained previously
	columns - categories of messages
	"""

	y_pred_test = model.predict(X_test)
	for idx, i in enumerate (y.columns.values):
		print(i)
		print(classification_report(y_test.values[idx], y_pred_test[idx]))

def save_model (model, clc_file):

	"""
	Saves the trained model in a pickle file

	Parameters:
	model - model trained previously
	clc_file - file path to save the model
	"""

	with open(clc_file, 'wb') as f:
		pickle.dump(model, f)


if __name__ == '__main__':

	database_file = sys.argv[1]
	clc_file = sys.argv[2]

	X, y = load_data (database_file)
	columns = y.columns.values

	X_train, X_test, y_train, y_test = train_test_split(X, y)

	model = train_model (X_train, y_train)
	eval_model(X_test, y_test, model, columns)
	save_model(model, clc_file)

