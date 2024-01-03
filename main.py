from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import streamlit as st

model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

def test_sentence(sentence: str, thresh_hold=0.2, debug=False):
    results = classifier(sentence)

    if debug:
        st.text('Debug is enabled, threshold value will be ignored\n')
        for result in results[0]:
            st.text(result)
        return

    may_include = []

    for result in results[0]:
        if result['score'] > thresh_hold:
            formatted_score = "{:.2f}".format(result['score'])
            may_include.append((result['label'], formatted_score))
    
    if may_include:
        for label, score in may_include:
            capitalized_label = label.capitalize()
            st.info(f'{capitalized_label} ({score})')
    else:
        st.info('Your sentence is totally fine')

st.title('Sentence Test App')
user_sentence = st.text_input('Enter your sentence to test it', value='I love NLP')
user_threshold = st.slider('Select threshold value', min_value=0.0, max_value=1.0, value=0.4, step=0.01)

if st.button('Test your sentence'):
    if user_sentence:
        st.success('Testing complete!')
        test_sentence(sentence=user_sentence, thresh_hold=user_threshold)
    else:
        st.error('Please enter a sentence before you test it!')