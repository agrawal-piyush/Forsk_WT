import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input , Output,State
import plotly.express as px
import pickle
import webbrowser
from sklearn.feature_extraction.text import TfidfTransformer as tfidfT, TfidfVectorizer as tfidfV

app = dash.Dash(__name__)# Creating Applicatiton 

def load_model():#loading model and data
    global dropdown,pickle_model,vocab 
    sdf = pd.read_csv('Scrappedreviews.csv')
    df = pd.read_csv('finaldata.csv')     
    df1 = df['Postivity']
    dropdown = sdf['reviews'].unique()
    
    file = open("pickle_model.pkl", 'rb')
    pickle_model = pickle.load(file) # Importing  model
    
    file1 = open("features.pkl", 'rb')
    vocab = pickle.load(file1) # Getting Vocabulary to predict given reviews

def open_browser(): #  Function to open Web-Browser 
    webbrowser.open_new( 'http://127.0.0.1:8050/' ) # Opening Browser

def create_ui():
    dropdown_style = { 'text-align' : 'center',
                       'margin-left' : '100px',
                       'width' : '80%' }
    H1style = html.style = { 'textAlign' : 'center', 
                             'background-color' : '#d3fcc5',
                             'color' : '#0091EA',
                             'border' : '6px solid #FF0266',  
                             'font-style' : 'italic',
                             'font-weight' : 'bold',
                             'padding-top' : '10px', 
                             'padding-bottom' : '10px' }
    
    TextAreaStyle = html.style = { 'border' : '6px solid #1DE9B6',
                                   'padding-top' : '15px',
                                   'width' : '100%',
                                   'padding-bottom' : '15px' }
    
    ButtonStyle = html.style = { 'border-color' : '6px solid blue',
                                 'background-color' : 'blue',
                                 'padding-top' : '10px',
                                 'width' : '200px',
                                 'padding-bottom' : '10px' }
    pie_style = { 'width' : '600px',
                  'height' : '450px' }

    pie_fig = px.pie( names = ['Positive Review','Negative Review'], 
                      values = [263527 , 263792], 
                      title = "BalencedReviews. csv data status " )


    app_layout = html.Div(html.Center([ 
        html.H1( id = 'Main_title', 
                children = 'Sentiments Analysis with Insights', 
                style = H1style ),
                                      
        html.Div([
            dcc.Graph(id='The_graph',
                     figure = pie_fig ,style = pie_style),
            ]),
        
        html.Div ([
            dcc.Dropdown(
                id = 'ddinput',
                options =[{'label': i, 'value': i} for i in dropdown],
                value='select',
                style =dropdown_style 
                ),
            html.Br(),
            html.Button('Check', id = 'ddbutton',n_clicks=0,style = ButtonStyle),
            html.Br(),
            html.Div(id = 'ddOutput',style={'whitespace':'pre-line'})
            ]),
            html.Br(), html.Br(),
        html.Div([
            dcc.Textarea(
                id ='textinput',
                value='enter the review',
                style = TextAreaStyle,
                ),
            html.Br(),
            html.Button ('Check',id ='textbutton',n_clicks=0,style =ButtonStyle),
            html.Br(),
            html.Div(id = 'textOutput',style={'whitespace':'pre-line'})
            ]),
        html.Br(), html.Br(),
        
    ])
        )
    
    return app_layout


def check_review(reviewText): # Function for Checking the User Entered Reviews
    transformer = tfidfT()
    loaded_vec = tfidfV( decode_error = "replace", vocabulary = vocab )
    reviewText = transformer.fit_transform( loaded_vec.fit_transform( [reviewText] ) )

    return pickle_model.predict(reviewText)


@app.callback(
    Output('ddOutput','children'),
    Input('ddbutton','n_clicks'),
    State('ddinput', 'value')
    )
def update_doutput(n_clicks,value):
    if n_clicks >0:
        print("value = ", str(value))
        
        result_list = check_review(value)
        
        if (result_list[0] == 0 ):
            result = "Negative Review "
    
        elif (result_list[0] == 1 ):
            result = "Positive Review "
    
        else:
            result = "Unknown"
        
        return result


@app.callback(
    Output('textOutput','children'),
    Input('textbutton','n_clicks'),
    State('textinput', 'value')
    )
def update_output(n_clicks,value):
    if n_clicks >0:
        
        print("value = ", str(value))
        
        result_list = check_review(value)
        
        if (result_list[0] == 0 ):
            result = "Negative Review"
    
        elif (result_list[0] == 1 ):
            result = "Positive Review "
    
        else:
            result = "Unknown"
        
        return result


def main():  # Main Function
    global app

    # Calling Fuctions
    load_model()
    open_browser()
    
    app.title = "Sentiments Analysis with Insights"  # Setting title of Web-Page
    app.layout = create_ui()  # Creating Layout of Application
    app.run_server()  # Starting / Running Server of Web-Page

    print("This would be executed only after the script is closed") 
    app = None  # Making all Global variables None


if __name__ == '__main__':  # Code to call main function
    main()