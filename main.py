from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import streamlit as st

model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

label_to_format = {
    'toxic': 'Toxic',
    'severe_toxic': 'Severe Toxic',
    'obscene' : 'Obscene',
    'threat' : 'Threat',
    'identity_hate' : 'Identity Hate',
    'insult' : 'Insult'
}

@st.cache_resource
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
            formatted_label = label_to_format[result['label']]
            may_include.append((formatted_label, formatted_score))
    
    if may_include:
        for label, score in may_include:
            capitalized_label = label.capitalize()
            st.info(f'{capitalized_label} ({score})')
    else:
        st.info('Your sentence is totally fine')

if __name__ == "__main__":
    st.title('Hate Speech Labeler')
    user_sentence = st.text_input('Enter your sentence to test it', value='I love NLP')
    user_threshold = st.number_input('Select threshold value', min_value=0.0, max_value=1.0, value=0.4)

    if st.button('Test your sentence'):
        if user_sentence:
            st.success('Testing complete!')
            test_sentence(sentence=user_sentence, thresh_hold=user_threshold)
        else:
            st.error('Please enter a sentence before you test it!')