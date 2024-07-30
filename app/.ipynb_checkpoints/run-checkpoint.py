import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


from flask import Flask
from flask import render_template, request, jsonify
from plotly import graph_objects 
import joblib
from sqlalchemy import create_engine

from wordcloud import WordCloud
import io
import base64



app = Flask(__name__)

def tokenize(text):
    """ Function to tokenize and lemmatize text.
    Args:
        text (str): Input text to be tokenized.
  
    Returns:
        clean_tokens (list): List of cleaned tokens.
    """
    
    
    # Normalize text by removing punctuation and converting to lowercase
    pun_regex = r"[^a-zA-Z0-9]"
    text = re.sub(pun_regex, " ", text)
    
    # Remove URLs from the text
    url_regex = (
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = re.sub(url_regex, " ", text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# Initiate count vectorizer
vect = CountVectorizer(tokenizer=tokenize, max_features=40, stop_words='english')
X = vect.fit_transform(df['message'].values)
top_feature_names = vect.get_feature_names_out()
feature_names_str = ' '.join(top_feature_names)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    # Sample data for the bar chart
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories_X = df.drop(columns=['id','message', 'original', 'genre'])
    categories_names = [col[:-2] for col in categories_X.mean().index]
    categories_mean = list(categories_X.mean().values*100)

    # Create the bar chart for genre distribution
    bar_chart1 = {
        'data': [
            graph_objects.Bar(
                x=genre_names,
                y=genre_counts
            )
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    }
    
    # Create the bar chart for category allocation distribution
    bar_chart2 = {
        'data': [
            graph_objects.Bar(
                x=categories_names,
                y=categories_mean
            )
        ],
        'layout': {
            'title': 'Proportion of responses by category',
            'yaxis': {
                'title': "Percentage (%)"
            },
            'xaxis': {
                'title': "Category"
            }
        }
    }
    
    
    # Generate the word cloud for top 40 responses
    wordcloud = WordCloud(
        width=800, height=400, background_color='white').generate(
        feature_names_str)
    
    # Convert the word cloud image to a format that Plotly can use
    img_stream = io.BytesIO()
    wordcloud.to_image().save(img_stream, format='PNG')
    img_stream.seek(0)
    
    # Encode the image as a base64 string
    img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
    
    # Create a Plotly figure with the word cloud image
    fig = graph_objects.Figure()
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{img_base64}',
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            xanchor='left',
            yanchor='top'
        )
    )
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        width=800,
        height=400,
        title="Top 40 words in disaster messages"
    )
    
    # Encode both Plotly graphs in JSON
    graphs = [
        {
            'data': bar_chart1['data'],
            'layout': bar_chart1['layout']
        },
        {
            'data': fig.to_dict()['data'],
            'layout': fig.to_dict()['layout']
        },
        {
            'data': bar_chart2['data'],
            'layout': bar_chart2['layout']
        }
    ]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with Plotly graphs
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()