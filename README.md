# SE464 Machine Learning Project

## Hate Speech Labeler

Streamlit app for hate speech detection using a fine-tuned BERT-based model. The model is trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset for multi-label classification.

- Code and data is available at this [notebook](https://www.kaggle.com/code/berkaysahiin/bert-fine-tune)

- The app is deployed and can be tested [here](https://huggingface.co/spaces/berkaysahiin/hataspeechlabeler) (also  available at this [link](https://huggingface.co/spaces/berkaysahiin/berkaysahiin-bert-base-uncased-jigsaw-toxic-classifier))

- The model is available at [hugging face](https://huggingface.co/berkaysahiin/bert-base-uncased-jigsaw-toxic-classifier)
 
 ![image](https://github.com/berkaysahiin/SE464/assets/92673021/9469b245-a567-4d35-8152-22acecf37e56)


 ## Local Installation 
   - Clone the repository:
     ```bash
     git clone https://github.com/berkaysahiin/SE464.git
   - Change into the directory:
     ```bash
     cd SE464
     ```
   - Virtual Environments:
     ```bash
     virtualenv venv
     .\venv\Scripts\activate
     ```
  - Requirements:
     ```bash
     pip install -r requirements.txt
     # if fails try before: pip install pipreqs && pipreqs 
     
     ```
     
   - Run the Streamlit app:
     ```bash
     streamlit run main.py

      ```   

## Model

- Data preprocessing involves cleaning text data, tokenization, and formatting for multi-label classification.

- The model is trained with TrainingArguments and Trainer from the Transformers library.

- Metrics such as F1 score, ROC AUC, and accuracy are used to evaluate the model's performance on the test set.
  
