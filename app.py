import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer

st.set_page_config(page_title='Spam SMS Detector ',layout='wide', initial_sidebar_state='expanded')




# st.markdown('<font color=‘red’>THIS TEXT WILL BE RED</font>', unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center;'>Spam SMS Classifier </h1>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.sidebar.columns([1, 2, 2, 1])
with col2:
    st.image('assets/sms.png', width=90)
with col3:
    st.image('assets/analyzer.png', width=90)

st.sidebar.caption(
    '**This is a simple machine learning web application which classify the given messages as Spam or Ham (Not Spam) built using NLP - Natural Language Processing.**')

st.sidebar.markdown('---') 

st.markdown("<h1 style='text-align: center; color: orange;'> Spam Detector for SMS ⚠️</h1>", unsafe_allow_html=True)



with st.expander('Click on the dropdown to see - How it works?'):
    st.subheader('Steps to Predict:')
    st.markdown(
        '1. Enter your SMS on text box.')
    st.markdown(
        '2. Click on Detect button for prediction.')
    st.markdown('')


st.sidebar.title('ML Model Details :')

st.sidebar.caption(
    '**Algorithm used:** **`Naive Bayes Classifier`**')
st.sidebar.caption(
    '**Dataset used:** **`UCI SMS Spam Collection`**')
st.sidebar.caption(
    '**Accuracy:** **`99.5%`**')


ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message :")

if st.button('Detect'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.error("Oh No! 😲 This is a SPAM Message.")
    else:
        st.success("Great! 😀 This is NOT a Spam Message.")



