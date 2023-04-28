import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_and_merge (csv1, csv2):

	"""
	Reads messages csv and categories csv and merge than

	Parameters
	csv1 - path to messages csv file
	csv2 - path to categories csv file

	Returns:
	df - merged data
	"""

	#read csvs
	messages = pd.read_csv(csv1)
	categories = pd.read_csv(csv2)

	#merge datasets
	df = messages.merge(categories, how='inner', on='id')

	return df 

def prepare (df):

	"""
	Prepare df

	Parameters
	df - csv with messages and categories merged

	Returns:
	df - processed data
	"""

	#Split categories into separate category columns
	categories = df['categories'].str.split(pat=';', expand=True)

	#Rename columns of categories with new column names.
	names = list(categories.iloc[0])
	names = [i.split('-')[0] for i in names]
	categories.set_axis(names, axis=1,inplace=True)

	#Convert category values to just numbers 0 or 1
	for i in names:
	  categories[i] = categories[i].str.slice(-1).astype(int)

	#Replace categories column in df with new category columns.
	df = df.drop(columns=['categories'])
	df = pd.concat([df, categories], axis=1)

	#Remove duplicates
	df = df.drop_duplicates()

	df[df['related'] == 2] = 1

	return df 

def save (df, database_file):

	"""
	Save the clean dataset into an sqlite database.

	Parameters:
	df - processed data
	database_file - path file to sqlite database
	"""

	engine = create_engine('sqlite:///' + database_file)
	df.to_sql('ELT_Preparation', engine, index=False)


if __name__ == '__main__':

	messages_file = sys.argv[1]
	categories_file = sys.argv[2]
	database_file = sys.argv[3]

	df = load_and_merge(messages_file, categories_file)
	df = prepare(df)
	save(df, database_file)