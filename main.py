import numpy as np
import pandas as pd

import string

import streamlit as st

header = st.beta_container()
dataset = st.beta_container()
feature = st.beta_container()
modelTraining = st.beta_container()
testing = st.beta_container()

# @st.cache
def get_data(file_name):
    df = pd.read_csv(file_name , header = None)
    return df

with header:
    st.title("Emotion Detection using Text")

with dataset:

    st.header("Emotion Detection Datasets")
    emotion_df = get_data("1-P-3-ISEAR.csv")
    emotion_df.columns = ['sn','Target','Sentence']
    emotion_df.drop('sn',inplace=True,axis =1)

    st.dataframe(emotion_df.head() )

    st.subheader("Lets check if the dataset is fairly distrributed.")

    col1 , col2 = st.beta_columns(2)

    target_count = emotion_df['Target'].value_counts()

    col1.table(target_count)
    col2.text("Line Chart of the total output counts")

    col2.line_chart(target_count )

    st.markdown("From the above data, we can easily say the data iss fairly distributed.")

with feature:
    st.header("Learning about Feature and converting them")
    
    #Converting the sentence to lowercase
    def lowercase(text):
        text = text.lower()
        return text
    emotion_df['Sentence'] = emotion_df['Sentence'].apply(lowercase)

    #Removing the Punction
    def remove_punct_num(text):
        text = "".join([char for char in text if char not in string.punctuation and not char.isdigit()])
        return text
    emotion_df['Sentence'] = emotion_df['Sentence'].apply(remove_punct_num)

    #Removing the stop words
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    def remove_stopwords(text):
        text = [w for w in text.split() if w not in stopwords.words('english')]
        return ' '.join(text)
    emotion_df['Sentence'] = emotion_df['Sentence'].apply(remove_stopwords)

    #Lemmatization i.e changing words into it's root form
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet')
    from nltk.corpus import wordnet    
    lemmatizer = WordNetLemmatizer()
    def lemmatize(text):
        text = [lemmatizer.lemmatize(word,'v') for word in text.split()]
        return ' '.join(text)
    emotion_df['Sentence'] = emotion_df['Sentence'].apply(lemmatize)

    st.markdown('As a data pre-processing, we have done the following things: -Converting the sentence to lowercase,-Removing the Punction , -Lemmatization i.e changing words into it is root form , -Removing the stop words')

    st.dataframe(emotion_df.head())

with modelTraining:
    st.header("time to train model")
    st.text("this is where we will train the model and rerun everything as per the hyperparameter tuining so on and so forth.")

    from sklearn.model_selection import train_test_split
    X = emotion_df['Sentence']
    y = emotion_df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=10)

    # st.text('TFIDF : It is technique to transform text into a meaningful vector of numbers. TFIDF penalizes words that come up too often and dont really have much use. So it rescales the frequency of words that are common which makes scoring more balanced')
   
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(X_train)
    test_tfidf = tfidf.transform(X_test)

    st.subheader('Model Building')

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_tfidf,y_train)

    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(train_tfidf,y_train)

    sel_col , disp_col = st.beta_columns(2)

    with sel_col:

        sel_col.header("Logistic Regression")

        sel_col.subheader("Logistic Regression Train Error")
        sel_col.write(lr.score(train_tfidf, y_train))

        sel_col.subheader("Logistic Regression Test Error")
        sel_col.write( lr.score(test_tfidf, y_test))
    
    with disp_col:

        disp_col.header("Naive Bias")

        disp_col.subheader("Naive Bias Train Error")
        disp_col.write(nb.score(train_tfidf, y_train))

        disp_col.subheader("Naive Bias Test Error")
        disp_col.write(nb.score(test_tfidf, y_test))

with testing:

    option = st.selectbox('Algorithm you would like to use?',
    ('LogisticRegression', 'Naive Bias'))

    input_feature = sel_col.text_input("Enter the statment to predict the  emotion type", "I am happy to help you.")
    
    test_sentence = tfidf.transform([input_feature])

    if(option == "Naive Bias"):
        y = nb.predict(test_sentence)
    else:
        y = lr.predict(test_sentence)
    
    st.write(y)