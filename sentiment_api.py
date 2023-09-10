from flask import Flask, jsonify, request 

from transformers import pipeline,  AutoTokenizer
from transformers import AutoModelForSequenceClassification
import requests  
import json 
import torch

CLASSIFIER_URL = "http://localhost:8002/classify"


app = Flask(__name__)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)



sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english" 
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name) 




def run_sentimental_model(sentences,token):
    new_sentences = [] 
    for sentence in sentences:
        if sentence.find(token) != -1:
            classifier = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer,device=device)  
            res = classifier(sentence)    
            if res[0]['label'] == "POSITIVE" and float(res[0]['score']) < 0.96:
                new_sentences.append(sentence)


    return new_sentences



def process_data(text,token): 
    print(type(text))
    text = text.replace('.', '.<eos>')
    text = text.replace('?', '?<eos>')
    text = text.replace('!', '!<eos>') 
    sentences = text.split('<eos>')  
    sentences =  [s for s in sentences if len(s) > 5]
    new_sen = run_sentimental_model(sentences,token) 
    if len(new_sen) == 0:
        return
    else:
        dataJson = {
            "sentences": new_sen,
            "company": token
        }

        json_data = json.dumps(dataJson)
        headers = {"Content-Type": "application/json"}
        res = requests.post(CLASSIFIER_URL, data=json_data, headers=headers) 
        if res.status_code == 200:
            print(res.text)
        else:
            print(f"POST request failed with status code: {res.status_code}")






@app.route('/ss', methods=['POST'])
def senti():

    file = request.files['file']  
    token = request.form['company'] 
    if file:
       file.save(f'{token}.txt')
    with open(f'{token}.txt', 'r', encoding='utf-8') as file:
        file_content = file.read()  


    process_data(file_content,token)

    result = {'message':  file_content }
    return result

    

app.run(port=8001)