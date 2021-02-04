import base64
import os
import random
import re
from os.path import join, dirname
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import fasttext
import fitz
import flask
import nltk
import plotly
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from app import app
import pandas as pd
import pandas as pd
from wordcloud import WordCloud

import dash_table

stop_words = stopwords.words('english')
# defined stopwords
newStopWords = ['created', 'modified', 'scout', 'strat', 'alun', 'rhydderch', 'description', 'tags', 'trends',
                'projects', 'page', 'steep', 'could']
stop_words.extend(newStopWords)
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()
# Beautifying names in the frontend dictionary
mapping_dict = {'locallifeandgreeneconomy': 'Local Life and Green Economy',
                'selfdrivingtransport': 'Self-Driving Transport',
                'alternativeenergysources': 'Alternative Energy Sources',
                '3dmobility': '3D Mobility',
                'smarthighways': 'Smart Highways',
                'sharingeconomyandsharedownership': 'Sharing economy and Shared ownership',
                'newmodesofpublictransport': 'New Modes of Public Transport',
                'futuremobility': 'Future Mobility',
                'virtualandaugmentedreality': 'Virtual and Augmented Reality',
                'aiandadvancedmachinelearning': 'AI and Advanced Machine Learning'}

# static path needs updation
project_root = dirname(dirname(__file__))
data_Path = join(project_root, 'assets')

UPLOAD_DIRECTORY = join(data_Path,'docs')
UPLOAD_DIRECTORY1 = join(data_Path,'multi')

########################################### Frontend ##########################################################
layout = html.Div(
    [

        dcc.ConfirmDialog(
            id='confirm-msg22',
        ),
        dcc.ConfirmDialog(
            id='confirm-msg33',
        ),

        html.Div(
            [

                html.Div(
                    [

                        html.Div(
                            [
                                html.H2("Future Scanner", style={"padding-left": "30px"}),
                                html.Div([
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),

                                    html.Div(
                                        [
                                            dcc.Upload(
                                                id="upload-data",
                                                children=html.Div([
                                                    # upload Button
                                                    html.A(
                                                        html.Button("Upload Document", style={'align-items': 'center'})
                                                    )

                                                ]

                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "60px",
                                                    "lineHeight": "60px",
                                                    "borderWidth": "1px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "5px",
                                                    "textAlign": "center",
                                                    "margin": "10px",
                                                }, multiple=True
                                            )],
                                        className="",
                                        style={"padding-left": "30px"}

                                    ),

                                    html.Br(),
                                    html.Br(),
                                    html.Div(
                                        [html.Button('Predict', id="Prediction",

                                                     style={
                                                         "width": "80%",
                                                         "height": "60px",
                                                         "lineHeight": "60px",
                                                         "borderWidth": "1px",
                                                         # "borderStyle": "dashed",
                                                         "borderRadius": "5px",
                                                         "textAlign": "center",
                                                         "margin": "10px",
                                                         "backgroundColor": "#00b0b9",
                                                         "color": "#fff"
                                                     }),

                                         ],
                                        id="",
                                        className="",
                                        style={"padding-left": "30px"}
                                    ),
                                    html.Div(
                                        dcc.Loading(id='loading-77',
                                                    children=
                                                    [html.H4(id="website_loader_22",
                                                             style={"backgroundColor": "#fff",
                                                                    'text-align': 'center', 'padding-top': '40px',
                                                                    'padding-left': '80px', }),
                                                     ], type='default', style={'display': 'hidden'}),
                                        id="",
                                        style={'text-align': 'center', 'font-weight': 'bold', 'padding-left': '40px'}),
                                    html.Div(
                                        dcc.Loading(id='loading-88',
                                                    children=
                                                    [html.H4(id="website_loader_33",
                                                             style={"backgroundColor": "#fff",
                                                                    'text-align': 'center', 'padding-top': '40px',
                                                                    'padding-left': '80px', }),
                                                     ], type='default', style={'display': 'hidden'}),
                                        id="",
                                        style={'text-align': 'center', 'font-weight': 'bold', 'padding-left': '40px'})

                                ],
                                    id="info-container",
                                    className="row container-display",

                                ),

                                html.Div([

                                    html.Iframe("Document Preview", id='Iframe_document',
                                                style={"width": "880px", "height": "715px"})],
                                    id="",
                                    className="pretty_container",
                                    style={"backgroundColor": "#e6d5f5"}
                                ),

                            ],
                            id="",
                            className="six columns",
                        ),

                        html.Div(
                            [

                                html.Div(
                                    [

                                        html.H3("Suggested Document Category ", style={"padding-left": "150px"}),
                                        html.Div(dash_table.DataTable(
                                            id='table-editing-simple10',
                                            style_cell={'textAlign': 'center', 'padding': '5px', 'fontSize': 18},
                                            editable=False
                                        ), style={"padding": "20px", "padding-bottom": "0px"}),
                                        html.Br(),

                                        html.Div(
                                            [
                                                html.H3("Word Cloud", style={"padding-left": "300px"}),
                                                html.Div(
                                                    [
                                                        dcc.Graph(id="rtagraph")
                                                    ],
                                                    className="",
                                                    # id="cross-filter-options-2",
                                                    style={"padding-left": "20px"}
                                                ),

                                            ],
                                            className="row flex-display",

                                        )

                                    ]
                                    ,
                                    id="",
                                    className="pretty_container",
                                    style={"backgroundColor": "#e6d5f5", "height": "750px"}
                                ),

                            ],
                            className="five columns",
                            style={"padding-top": "160px", "height": "650px"}

                        ),
                        html.Div(
                            [
                                html.Div([
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),

                                    html.Div(
                                        [
                                            dcc.Upload(
                                                id="upload-data3",
                                                children=html.Div([
                                                    html.A(
                                                        html.Button("Bulk Upload", style={'align-items': 'center'})
                                                    )

                                                ]

                                                ),
                                                style={
                                                    "width": "100%",
                                                    "height": "60px",
                                                    "lineHeight": "60px",
                                                    "borderWidth": "1px",
                                                    "borderStyle": "dashed",
                                                    "borderRadius": "5px",
                                                    "textAlign": "center",
                                                    "margin": "10px",
                                                }, multiple=True
                                            )],
                                        className="",
                                        style={"padding-left": "30px"}

                                    ),

                                    html.Br(),
                                    html.Br(),
                                    html.Div(
                                        [html.Button('Evaluate', id="Evaluate",

                                                     style={
                                                         "width": "80%",
                                                         "height": "60px",
                                                         "lineHeight": "60px",
                                                         "borderWidth": "1px",
                                                         "borderRadius": "5px",
                                                         "textAlign": "center",
                                                         "margin": "10px",
                                                         "backgroundColor": "#00b0b9",
                                                         "color": "#fff"
                                                     }),

                                         ],
                                        id="",
                                        className="",
                                        style={"padding-left": "30px"}
                                    ),
                                    html.Div(
                                        dcc.Loading(id='loading-66',
                                                    children=
                                                    [html.H4(id="website_loader_11",
                                                             style={"backgroundColor": "#fff",
                                                                    'text-align': 'center', 'padding-top': '40px',
                                                                    'padding-left': '80px', }),
                                                     ], type='default', style={'display': 'hidden'}),
                                        id="",
                                        style={'text-align': 'center', 'font-weight': 'bold', 'padding-left': '40px'}),
                                    html.Div(
                                        dcc.Loading(id='loading-06',
                                                    children=
                                                    [html.H4(id="website_loader_44",
                                                             style={"backgroundColor": "#fff",
                                                                    'text-align': 'center', 'padding-top': '40px',
                                                                    'padding-left': '80px', }),
                                                     ], type='default', style={'display': 'hidden'}),
                                        id="",
                                        style={'text-align': 'center', 'font-weight': 'bold', 'padding-left': '40px'})

                                ],
                                    id="info-container-1",
                                    className="row container-display",

                                ),

                                html.Div([

                                    html.Div(id='dataframe_output', style={"padding": "20px"})

                                ],
                                    id="",
                                    className="pretty_container",
                                    style={"backgroundColor": "#e6d5f5"}
                                ),

                            ],
                            id="",
                            className="eleven columns",
                        ),

                    ],
                    className="row flex-display",
                ),

            ]),

    ],
    style={"backgroundColor": "#fff", "height": "332px"}
)


########################################### Backend ##########################################################

@app.callback(
    [Output("confirm-msg33", "message"), Output("confirm-msg33", "displayed"), Output("website_loader_11", "children")],
    [Input("upload-data3", "filename"), Input("upload-data3", "contents")],
)
def update_multi_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        if uploaded_filenames[0][-3:] == 'pdf':
            file_name = ""
            for name, data in zip(uploaded_filenames, uploaded_file_contents):
                save_file_multi(name, data)
                file_name = name
            return 'Document is Uploaded Successfully', True, ""
        else:
            return 'Incorrect File Format,Required Format is .pdf file', True, ""

html.P
@app.callback(
    [Output("confirm-msg22", "message"), Output("confirm-msg22", "displayed"), Output('Iframe_document', 'src'),
     Output("website_loader_22", "children")],
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        if uploaded_filenames[0][-3:] == 'pdf':
            file_name = ""
            for name, data in zip(uploaded_filenames, uploaded_file_contents):
                save_file(name, data)
                file_name = name
            return 'Document is Uploaded Successfully', True, app.get_asset_url("docs//" + str(file_name)), ""
        else:
            return 'Incorrect File Format,Required Format is .pdf file', True, None, ""


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def save_file_multi(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY1, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def get_bulk_dataframe():
    files = os.listdir(UPLOAD_DIRECTORY1)
    global mapping_dict

    File_Name = []

    for file in files:
        File_Name.append(str(UPLOAD_DIRECTORY1) + str("\\") + file)
    print(File_Name)
    data_frame = pd.DataFrame(zip(File_Name), columns=["File Name"])

    def pdf_reader(rows):
        page_text = ''
        doc = fitz.open(rows)
        for i in range(doc.pageCount):
            page = doc.loadPage(i)
            page_str = page.getText("text")
            page_text = page_text + page_str
        return page_text

    data_frame["Raw_Data"] = data_frame["File Name"].apply(pdf_reader)
    data_frame["File name"] = data_frame["File Name"].apply(lambda x: x.split('\\')[-1][:-4])

    def preprocessing(rows):

        text = rows.lower()
        # /*********************Remove number*******************/
        text = re.sub(r'\d+', ' ', text)
        # /*****************Remove Punctuation****************/
        text = re.sub(r'[^\w\s]', ' ', text)
        # /*****************Remove \xa0****************/
        text = re.sub(r'\xa0', '', text)
        # /*****************Remove \x0c****************/
        text = re.sub(r'\x0c', '', text)
        #    /**********Common word removal************/

        #    /*****************Remove stop words************/
        token_text = word_tokenize(text)
        tokens_without_sw = [word for word in token_text if not word in stop_words]
        text_stem = [ps.stem(word) for word in tokens_without_sw]
        text = (" ").join(text_stem)
        # /***************Remove space line character*********/
        text = text.replace('\n', ' ')
        #    /********************Remove duplicate space**********/
        text = " ".join(text.split())
        return text

    data_frame["Clean_Text"] = data_frame["Raw_Data"].apply(preprocessing)

    def wc(text, tfidf_vectorizer):
        doc_text = [text]
        tf_model = tfidf_vectorizer.fit_transform(doc_text)
        text_scored = tfidf_vectorizer.transform(doc_text)  # note - .fit() not .fit_transform
        terms = tfidf_vectorizer.get_feature_names()
        scores = text_scored.toarray().flatten().tolist()
        data = list(zip(terms, scores))
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
        top_words = sorted_data[:2]
        final_words = []
        for i in range(len(top_words)):
            final_words.append(top_words[i][0])
        return final_words

    def clean(rows):
        # /*********************Convert text into lower case**********/
        text = rows.lower()
        # /*********************Remove number*******************/
        text = re.sub(r'\d+', ' ', text)
        # /*****************Remove Punctuation****************/
        text = re.sub(r'[^\w\s]', ' ', text)
        # /*****************Remove \xa0****************/
        text = re.sub(r'\xa0', '', text)
        # /*****************Remove \x0c****************/
        text = re.sub(r'\x0c', '', text)
        #    /*****************Remove stop words************/
        token_text = word_tokenize(text)
        tokens_without_sw = [word for word in token_text if not word in stop_words]
        text_lem = [wn.lemmatize(word) for word in tokens_without_sw]
        text = (" ").join(text_lem)
        # /***************Remove space line character*********/
        text = text.replace('\n', ' ')
        #    /********************Remove duplicate space**********/
        text = " ".join(text.split())
        tfidf_vectorizer = TfidfVectorizer()
        top_unigram_words = wc(text, tfidf_vectorizer)
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
        top_bigram_words = wc(text, tfidf_vectorizer)
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 3))
        top_trigram_words = wc(text, tfidf_vectorizer)
        combined_gram = top_trigram_words + top_bigram_words + top_unigram_words
        return ", ".join(combined_gram)

    data_frame["Document Features"] = data_frame["Raw_Data"].apply(clean)

    # /*********************Predections*****************/
    # static path needs updation
    model = fasttext.load_model(join(project_root,'RTA_Future_Scanner.bin'))

    data_frame["predicted_label_1"] = data_frame["Clean_Text"].apply(lambda x: model.predict(x, k=-1)[0][0])

    data_frame["predicted_label_1_probab"] = data_frame["Clean_Text"].apply(lambda x: model.predict(x, k=-1)[1][0])

    data_frame["predicted_label_2"] = data_frame["Clean_Text"].apply(lambda x: model.predict(x, k=-1)[0][1])

    data_frame["predicted_label_2_probab"] = data_frame["Clean_Text"].apply(lambda x: model.predict(x, k=-1)[1][1])

    data_frame["predicted_label_1"] = data_frame["predicted_label_1"].apply(
        lambda x: x.replace("__label__", '').replace("__n_", ''))

    data_frame["predicted_label_2"] = data_frame["predicted_label_2"].apply(
        lambda x: x.replace("__label__", '').replace("__n_", ''))

    data_frame["predicted_label_1_probab"] = data_frame["predicted_label_1_probab"].apply(
        lambda x: str(round(x * 100, 2)) + "%")

    data_frame["predicted_label_2_probab"] = data_frame["predicted_label_2_probab"].apply(
        lambda x: str(round(x * 100, 2)) + "%")

    data_frame["predicted_label_1"] = data_frame["predicted_label_1"].apply(
        lambda x: " ".join(re.findall('[3]*[A-Z][a-z]*', x)))

    data_frame["predicted_label_1"] = data_frame["predicted_label_1"].apply(
        lambda x: "".join(str(x).lower().split(" "))).replace(mapping_dict)

    data_frame["predicted_label_2"] = data_frame["predicted_label_2"].apply(
        lambda x: " ".join(re.findall('[3]*[A-Z][a-z]*', x)))

    data_frame["predicted_label_2"] = data_frame["predicted_label_2"].apply(
        lambda x: "".join(str(x).lower().split(" "))).replace(mapping_dict)

    data_frame["Label 1"] = data_frame["predicted_label_1"]
    data_frame["Probability 1"] = data_frame["predicted_label_1_probab"]
    data_frame["Label 2"] = data_frame["predicted_label_2"]
    data_frame["Probability 2"] = data_frame["predicted_label_2_probab"]
    final_data = data_frame.loc[:,
                 ["File name", "Document Features", "Label 1", "Probability 1", "Label 2", "Probability 2"]]

    return final_data


@app.callback(
    [Output('dataframe_output', 'children'), Output("website_loader_44", "children")]
    ,
    [Input('Evaluate', 'n_clicks')])
def show_summary(n_clicks):
    info_dataframe = get_bulk_dataframe()
    global mapping_dict

    data = info_dataframe.to_dict("rows")
    cols = [{"name": i, "id": i} for i in info_dataframe.columns]

    child = html.Div([
        dash_table.DataTable(
            id='table',
            data=data,
            columns=cols,
            style_cell={'width': '50px',
                        'height': '30px',
                        'textAlign': 'center',
                        'font-size': '20px'},
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    ])
    for filename in os.listdir(UPLOAD_DIRECTORY1):
        file_path = os.path.join(UPLOAD_DIRECTORY1, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return child, ""


def word_cloud():

    image_filename = join(data_Path,'wc.png')
    plotly_logo = base64.b64encode(open(image_filename, 'rb').read())
    fig = go.Figure()
    img_width = 1450
    img_height = 700
    scale_factor = 0.5
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        scaleanchor="x"
    )
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source='data:image/png;base64,{}'.format(plotly_logo.decode())))
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig


@app.callback([

    Output("table-editing-simple10", "data"),
    Output("table-editing-simple10", "columns"),
    Output("rtagraph", "figure"),
    Output("website_loader_33", "children")
],
    [Input('Prediction', 'n_clicks'), Input("upload-data", "filename")])
def prediction(n_clicks, uploaded_filenames):
    threshold = 0.80
    global mapping_dict
    print(n_clicks, uploaded_filenames)
    if n_clicks is not None and uploaded_filenames is not None and threshold is not None:
        page_text = ''
        doc = fitz.open(".\\assets\\docs\\" + uploaded_filenames[0])
        print(doc.pageCount)
        for i in range(doc.pageCount):
            page = doc.loadPage(i)
            page_str = page.getText("text")
            page_text = page_text + page_str
        text = page_text.lower()

        # /*********************Remove number*******************/
        text = re.sub(r'\d+', ' ', text)
        # /*****************Remove Punctuation****************/
        text = re.sub(r'[^\w\s]', ' ', text)
        # /*****************Remove \xa0****************/
        text = re.sub(r'\xa0', '', text)
        # /*****************Remove \x0c****************/
        text = re.sub(r'\x0c', '', text)
        #    /*****************Remove stop words************/
        token_text = word_tokenize(text)
        tokens_without_sw = [word for word in token_text if not word in stop_words]
        text_stem = [ps.stem(word) for word in tokens_without_sw]
        text = (" ").join(text_stem)
        # /***************Remove space line character*********/
        text = text.replace('\n', ' ')
        # /********************Remove duplicate space**********/
        text = " ".join(text.split())  # /**********Common word removal************/

        model = fasttext.load_model(join(project_root,'RTA_Future_Scanner.bin'))
        predicted_label_1 = model.predict(text, k=-1)[0][0]
        predicted_label_1_probab = model.predict(text, k=-1)[1][0]

        predicted_label_2 = model.predict(text, k=-1)[0][1]
        predicted_label_2_probab = model.predict(text, k=-1)[1][1]
        predicted_label_3 = model.predict(text, k=-1)[0][2]
        predicted_label_3_probab = model.predict(text, k=-1)[1][2]
        predicted_label_4 = model.predict(text, k=-1)[0][3]
        predicted_label_4_probab = model.predict(text, k=-1)[1][3]

        predicted_label_1 = predicted_label_1.replace("__label__", '').replace("__n_", '')
        predicted_label_2 = predicted_label_2.replace("__label__", '').replace("__n_", '')
        predicted_label_3 = predicted_label_3.replace("__label__", '').replace("__n_", '')
        predicted_label_4 = predicted_label_4.replace("__label__", '').replace("__n_", '')

        predicted_label_1 = " ".join(re.findall('[3]*[A-Z][a-z]*', predicted_label_1))
        predicted_label_1 = mapping_dict.get("".join(predicted_label_1.lower().split(" ")), predicted_label_1)
        predicted_label_2 = " ".join(re.findall('[3]*[A-Z][a-z]*', predicted_label_2))
        predicted_label_2 = mapping_dict.get("".join(predicted_label_2.lower().split(" ")), predicted_label_2)
        predicted_label_3 = " ".join(re.findall('[3]*[A-Z][a-z]*', predicted_label_3))
        predicted_label_3 = mapping_dict.get("".join(predicted_label_3.lower().split(" ")), predicted_label_3)
        predicted_label_4 = " ".join(re.findall('[3]*[A-Z][a-z]*', predicted_label_4))
        predicted_label_4 = mapping_dict.get("".join(predicted_label_4.lower().split(" ")), predicted_label_4)

        Confidence_Score = '-'
        j1 = ""
        j2 = 0
        j3 = ""
        j4 = 0
        j5 = ""
        j6 = 0

        if predicted_label_1_probab >= threshold:
            Sample1 = []
            Sample1.append(predicted_label_1)
            Sample1.append(predicted_label_1_probab)
            j1 = (Sample1[0])
            j2 = round(Sample1[1], 2)
        elif (predicted_label_1_probab + predicted_label_2_probab) >= threshold:
            Sample1 = []
            Sample1.append(predicted_label_1)
            Sample1.append(predicted_label_1_probab)
            Sample2 = []
            Sample2.append(predicted_label_2)
            Sample2.append(predicted_label_2_probab)
            j1 = (Sample1[0])
            j2 = round(Sample1[1], 2)
            j3 = Sample2[0]
            j4 = round(Sample2[1], 2)
        elif (predicted_label_1_probab + predicted_label_2_probab + predicted_label_3_probab) >= threshold:
            Sample1 = []
            Sample1.append(predicted_label_1)
            Sample1.append(predicted_label_1_probab)
            Sample2 = []
            Sample2.append(predicted_label_2)
            Sample2.append(predicted_label_2_probab)
            Sample3 = []
            Sample3.append(predicted_label_3)
            Sample3.append(predicted_label_3_probab)
            j1 = (Sample1[0])
            j2 = round(Sample1[1], 2)
            j3 = Sample2[0]
            j4 = round(Sample2[1], 2)
            j5 = Sample3[0]
            j6 = round(Sample3[1], 2)
        else:
            j1 = '-'
            j2 = '-'
            j3 = '-'
            j4 = '-'
            j5 = '-'
            j6 = '-'
        j2 = str(j2 * 100) + " %" if j2 != "-" else str(j2)
        j4 = str(j4 * 100) + " %" if j4 != "-" else str(j4)
        j6 = str(j6 * 100) + " %" if j6 != "-" else str(j6)

        text_lem = [wn.lemmatize(word) for word in tokens_without_sw]
        word_text = (" ").join(text_lem)
        # /***************Remove space line character*********/
        word_text = word_text.replace('\n', ' ')
        #    /********************Remove duplicate space**********/
        word_text = " ".join(word_text.split())
        tfidf_vectorizer = TfidfVectorizer()
        top_unigram_words = wc(word_text, tfidf_vectorizer)
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
        top_bigram_words = wc(word_text, tfidf_vectorizer)
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(3, 3))
        top_trigram_words = wc(word_text, tfidf_vectorizer)
        test = []
        test = top_unigram_words
        # + top_bigram_words + top_trigram_words
        d = {}
        for a, x in test:
            d[a] = 10 * x

        wordcloud = WordCloud(width=1450, height=700, background_color='white')
        wordcloud.generate_from_frequencies(frequencies=d)
        wordcloud.to_file(join(data_Path,'wc.png'))

        # words = list(d.keys())
        # weights = [round(each) for each in list(d.values())]
        wordcloud_fig = word_cloud()
        df = {
            "Prediction": [j1, j3, j5],
            "Probability": [j2, j4, j6],
        }

        df = pd.DataFrame(df)
        df = df[df['Prediction'] != ""]
        # fig.show()
        print("reached here=====================================================1")

        return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], wordcloud_fig, ""


def wc(text, tfidf_vectorizer):
    doc_text = [text]
    tf_model = tfidf_vectorizer.fit_transform(doc_text)
    text_scored = tfidf_vectorizer.transform(doc_text)  # note - .fit() not .fit_transform
    terms = tfidf_vectorizer.get_feature_names()
    scores = text_scored.toarray().flatten().tolist()
    data = list(zip(terms, scores))
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    top_words = sorted_data[:15]
    return top_words
