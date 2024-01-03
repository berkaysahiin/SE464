# SE464 Machine Learning Project

## Hate Speech Labeler

Streamlit app for hate speech detection using a fine-tuned BERT-based model. The model is trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset for multi-label classification.

- The app is deployed and accessible [here.](https://hatespeechlabeler.streamlit.app)
- Code and data is available at this [notebook](https://www.kaggle.com/code/berkaysahiin/bert-fine-tune).

 ## Local Installation 
   - Clone the repository:
     ```bash
     git clone https://github.com/berkaysahiin/SE464.git
   - Change into the directory:
     ```bash
     cd SE464
     ```
   - Virtual Environment and Dependencies:
     ```bash
     virtualenv venv
     ./venv/bin/activate 
     pip install -r requirements.txt
     ```
   - Run the Streamlit app:
     ```bash
     streamlit run main.py

      ```   

## Model

- Data preprocessing involves cleaning text data, tokenization, and formatting for multi-label classification.

- The model is trained with TrainingArguments and Trainer from the Transformers library.

- Metrics such as F1 score, ROC AUC, and accuracy are used to evaluate the model's performance on the test set.

## Development Environment

- Python version: 3.8.5
- Streamlit version: 1.2.0
- Transformers version: (Specify the version you used for fine-tuning)
