# /**********************Import all packages*******************/
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
# pip install PyMuPDF
import fitz
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import fasttext
import csv
import random
from os.path import join, dirname


# /************* Use the NLTK Downloader to obtain the resource *************/

nltk.download('wordnet')

# /*****************************Set the following paths*******************/

path = dirname(dirname(__file__))
Output_Path = join(join(path, 'Data'), 'Output')  # Output data path
File1_Path = join(join(path, 'Data'), 'Files1')  # Output data path
File2_Path = join(join(path, 'Data'), 'Files2')  # Output data path

# /**********************Load File set 1******************/
# /*********************Place all the files from File1 folder into File1_Path*********/
# pdf_reader function reads all the documents/articles in pdf format and combine in one dataframe.

os.chdir(File1_Path)

files = os.listdir(File1_Path)

Key_Trend = []
Trend = []
File_Name = []

for file in files:
    Key_Trend.append(file.split('_')[0].replace('.pdf', ''))
    Trend.append(file.split('_')[-1][:-4])
    File_Name.append(file)

file1 = pd.DataFrame(zip(Key_Trend, Trend, File_Name), columns=["Label", "Sub_Trend", "File_Name"])


def pdf_reader(rows):
    page_text = ''
    doc = fitz.open(rows)
    for i in range(doc.pageCount):
        page = doc.loadPage(i)
        page_str = page.getText("text")
        page_text = page_text + page_str
    return page_text


file1["Raw_Data"] = file1["File_Name"].apply(pdf_reader)

file1 = file1.loc[:, ["Label", "File_Name", "Raw_Data"]]

# /**********************Load File set 2******************/
# /*********************Place all the files from File2 folder into File2_Path*********/
# pdf_reader function reads all the documents/articles in pdf format and combine in one dataframe.
os.chdir(File2_Path)

files = os.listdir(File2_Path)

Key_Trend1 = []
Key_Trend2 = []
Key_Trend3 = []
Key_Trend4 = []
File_Name = []

for file in files:
    file_list = file.split('__')
    if len(file_list) == 2:
        Key_Trend1.append(file_list[0])
        Key_Trend2.append('')
        Key_Trend3.append('')
        Key_Trend4.append('')
        File_Name.append(file)
    elif len(file_list) == 3:
        Key_Trend1.append(file_list[0])
        Key_Trend2.append(file_list[1])
        Key_Trend3.append('')
        Key_Trend4.append('')
        File_Name.append(file)
    elif len(file_list) == 4:
        Key_Trend1.append(file_list[0])
        Key_Trend2.append(file_list[1])
        Key_Trend3.append(file_list[2])
        Key_Trend4.append('')
        File_Name.append(file)
    elif len(file_list) == 5:
        Key_Trend1.append(file_list[0])
        Key_Trend2.append(file_list[1])
        Key_Trend3.append(file_list[2])
        Key_Trend4.append(file_list[3])
        File_Name.append(file)

file2 = pd.DataFrame(zip(Key_Trend1, Key_Trend2, Key_Trend3, Key_Trend4, File_Name),
                     columns=["Label1", "Label2", "Label3", "Label4", "File_Name"])

df1 = file2.loc[:, ["Label1", "File_Name"]]

df1.rename(columns={"Label1": "Label"}, inplace=True)

df2 = file2.loc[file2["Label2"] != '', ["Label2", "File_Name"]]

df2.rename(columns={"Label2": "Label"}, inplace=True)

df3 = file2.loc[file2["Label3"] != '', ["Label3", "File_Name"]]

df3.rename(columns={"Label3": "Label"}, inplace=True)

df4 = file2.loc[file2["Label4"] != '', ["Label4", "File_Name"]]

df4.rename(columns={"Label4": "Label"}, inplace=True)

file2 = df1.append([df2, df3, df4], ignore_index=True)


def pdf_reader(rows):
    page_text = ''
    doc = fitz.open(rows)
    for i in range(doc.pageCount):
        page = doc.loadPage(i)
        page_str = page.getText("text")
        page_text = page_text + page_str
    return page_text


file2["Raw_Data"] = file2["File_Name"].apply(pdf_reader)

base_data = file1.append([file2], ignore_index=True)

# /****************************Oversampling to train the model**************/
# /***************************Oversampling is done for imbalance class********/
max_length = 60


def oversampling(rows):
    global max_length
    df = base_data.loc[base_data["Label"] == rows["Label"], :]
    ratio = len(df) / max_length
    if ratio <= 0.40:
        token_text = word_tokenize(rows["Raw_Data"])
        random.shuffle(token_text)
        text = (" ").join(token_text)
        return text
    else:
        return ""


wn = nltk.WordNetLemmatizer()

base_data["Oversample_Text"] = base_data[["Label", "Raw_Data"]].apply(oversampling, axis=1)

os_df = base_data.loc[base_data["Oversample_Text"] != "", ["Label", "File_Name", "Oversample_Text"]]

os_df.rename(columns={"Oversample_Text": "Raw_Data"}, inplace=True)

base_data.drop(["Oversample_Text"], inplace=True, axis=1)

base_data = base_data.append([os_df], ignore_index=True)

base_data_v1 = base_data.loc[~(base_data["Label"].isin(
    ["Novel Payment Systems", "Biometrics and Human-Machine Interface", "n_Self-Driving Transport"])), :]

base_data_v1["Label"] = base_data_v1["Label"].apply(lambda x: x.replace(' ', ''))

base_data_v1['Label'] = ['__label__' + s for s in base_data_v1['Label']]

base_data_v1 = base_data_v1.groupby(['Raw_Data'])['Label'].apply(' '.join).reset_index()

file_name = base_data.drop_duplicates(subset=['Raw_Data'])

base_data_v2 = base_data_v1.merge(file_name[['Raw_Data', "File_Name"]], left_on='Raw_Data', right_on='Raw_Data',
                                  how='left')


# /****************************Finding Document Features*******************/
# wc function is used to extract document features based on n-gram model and TF scores.
def wc(text, tfidf_vectorizer):
    doc_text = [text]
    tf_model = tfidf_vectorizer.fit_transform(doc_text)
    text_scored = tfidf_vectorizer.transform(doc_text)
    terms = tfidf_vectorizer.get_feature_names()
    scores = text_scored.toarray().flatten().tolist()
    data = list(zip(terms, scores))
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    top_words = sorted_data[:2]
    final_words = []
    for i in range(len(top_words)):
        final_words.append(top_words[i][0])
    return final_words


nltk.download("stopwords")
stop_words = stopwords.words('english')
newStopWords = ['created', 'modified', 'scout', 'strat', 'alun', 'rhydderch', 'description', 'tags', 'trends',
                'projects', 'page', 'steep', ]
stop_words.extend(newStopWords)
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()


# /********************************Data Preprocessing*****************/
# preprocessing function is used to cleansing the Raw data and will generate Clean_Text and Document_Features
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
    #    /*****************Remove stop words************/
    token_text = word_tokenize(text)
    tokens_without_sw = [word for word in token_text if not word in stop_words]
    ##    text_lem = [wn.lemmatize(word) for word in tokens_without_sw]
    text_stem = [ps.stem(word) for word in tokens_without_sw]
    text = (" ").join(text_stem)
    # /***************Remove space line character*********/
    text = text.replace('\n', ' ')
    #    /********************Remove duplicate space**********/
    text = " ".join(text.split())
    # /********************************For word cloud**********************/
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
    combined_gram = top_unigram_words + top_bigram_words + top_trigram_words
    return text, combined_gram


base_data_v2["Clean_Text"], base_data_v2["Document_Features"] = zip(*base_data_v2["Raw_Data"].apply(preprocessing))

# /********************************Model Building*************************/
# /*************We are using Fasttext to build our model*******************/

train_data = base_data_v2.loc[:, ["Label", "Clean_Text"]]

train_data.to_csv(Output_Path + '\\train_data.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE,
                  quotechar="", escapechar=" ")

model = fasttext.train_supervised(input=Output_Path + '\\train_data.txt', lr=0.5, epoch=100, loss='hs')


# /********************************Model Predictions***********************/

def predictions(rows):
    predicted_label_1 = model.predict(rows, k=-1)[0][0]
    predicted_label_1_probab = model.predict(rows, k=-1)[1][0]
    predicted_label_2 = model.predict(rows, k=-1)[0][1]
    predicted_label_2_probab = model.predict(rows, k=-1)[1][1]
    try:
        predicted_label_3 = model.predict(rows, k=-1)[0][2]
        predicted_label_3_probab = model.predict(rows, k=-1)[1][2]
    except:
        predicted_label_3 = ''
        predicted_label_3_probab = ''
    if predicted_label_1_probab >= 0.80 or predicted_label_2_probab < 0.15:
        return predicted_label_1, predicted_label_1_probab, ' ', ' ', ' ', ' '
    elif (predicted_label_1_probab + predicted_label_2_probab) >= 0.70:
        return predicted_label_1, predicted_label_1_probab, predicted_label_2, predicted_label_2_probab, ' ', ' '
    else:
        return predicted_label_1, predicted_label_1_probab, predicted_label_2, predicted_label_2_probab, predicted_label_3, predicted_label_3_probab


base_data_v2["predicted_label_1"], base_data_v2["predicted_label_1_probab"], base_data_v2["predicted_label_2"], \
base_data_v2["predicted_label_2_probab"], base_data_v2["predicted_label_3"], base_data_v2[
    "predicted_label_3_probab"] = zip(*base_data_v2["Clean_Text"].apply(predictions))


# /****************This function is used to clean the model output*************/
def clean(rows):
    text = rows.replace("__label__", '')
    return text


base_data_v2["Label"] = base_data_v2["Label"].apply(clean)
base_data_v2["predicted_label_1"] = base_data_v2["predicted_label_1"].apply(clean)
base_data_v2["predicted_label_2"] = base_data_v2["predicted_label_2"].apply(clean)
base_data_v2["predicted_label_3"] = base_data_v2["predicted_label_3"].apply(clean)

base_data_v2.rename(columns={"Label": "Actual_Label"}, inplace=True)

base_data_v3 = base_data_v2.loc[:,
               ["File_Name", "Actual_Label", "Document_Features", "predicted_label_1", "predicted_label_1_probab",
                "predicted_label_2", "predicted_label_2_probab", "predicted_label_3", "predicted_label_3_probab"]]

# /*****************Export the output results*****************/
base_data_v3.to_excel(Output_Path + '\\Scoring_Data.xlsx')
