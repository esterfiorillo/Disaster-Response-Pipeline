import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/ELT_Preparation.db')
df = pd.read_sql_table('ELT_Preparation', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    df_news = df[df['genre'] == 'news']
    df_direct = df[df['genre'] == 'direct']
    df_social = df[df['genre'] == 'social']

    # category data for plotting
    categories_news =  df_news[df_news.columns[4:]]
    cate_counts_news = (categories_news.mean()*categories_news.shape[0])
    cate_names_news = list(cate_counts_news.index)

    # category data for plotting
    categories_direct =  df_direct[df_direct.columns[4:]]
    cate_counts_direct = (categories_direct.mean()*categories_direct.shape[0])
    cate_names_direct = list(cate_counts_direct.index)

    # category data for plotting
    categories_social =  df_social[df_social.columns[4:]]
    cate_counts_social = (categories_social.mean()*categories_social.shape[0])
    cate_names_social = list(cate_counts_social.index)



    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cate_names_news,
                    y=cate_counts_news
                )
            ],

            'layout': {
                'title': 'Categories Distribution in News Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        # category plotting (Visualization#2)
        {
            'data': [
                Bar(
                    x=cate_names_direct,
                    y=cate_counts_direct
                )
            ],

            'layout': {
                'title': 'Categories Distribution in Direct Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
        },
        # Categories Distribution in Direct Genre (Visualization#3)
        {
            'data': [
                Bar(
                    x=cate_names_social,
                    y=cate_counts_social
                )
            ],

            'layout': {
                'title': 'Categories Distribution in Social Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()