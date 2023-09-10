
import requests
import copy
import torch
from transformers import pipeline,  AutoTokenizer
from transformers import AutoModelForSequenceClassification 
import openai 
from bs4 import BeautifulSoup 
import json 
from flask import Flask, request, jsonify




app = Flask(__name__)   


sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english" 
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)   
classifier = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)  



classifier2 = pipeline("zero-shot-classification")  
LABELS = ["historic", "financial" ,  "products", "services", "general activity", "business model", "areas of expertise",  "news"] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 






def google_search(query): 
    
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.text
    else:
        return None 
    

def extract_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    links = []
    
    for link in soup.find_all('a'):
        links.append(link)
    
    return links  



def parse_url(url,token):
    response = requests.get(url) 

    if response.status_code == 200:
            response = requests.get(url)
            linked_page_soup = BeautifulSoup(response.text, 'html.parser')              
            linked_page_text = linked_page_soup.get_text() 
            file_path = "example.txt" 
            linked_page_text = " ".join(linked_page_text.split())

            # Open the file in write mode and write the content
            with open(file_path, 'a',encoding='utf-8') as file: 
                # Append the additional content to the end of the file
                file.write('\n\n' + linked_page_text)  




def parse_urls_if_exist(links,company): 
    company = company.lower()
    for link in links:
        href = link.get('href')
        if href and href.startswith('/url?'):
            url_start = href.find('url=') + 4
            url_end = href.find('&', url_start)
            if url_start != -1 and url_end != -1:
                url = href[url_start:url_end] 
                if url.find(company) != -1: #and url.find("about")  != -1: -- domain/about cases
                    parse_url(url,company)             
        else:
            pass 



def run_sentimental_model(sentences,token):
    new_sentences = [] 
    for sentence in sentences:
        if sentence.find(token) != -1:
            res = classifier(sentence)    
            if res[0]['label'] == "POSITIVE" and float(res[0]['score']) < 0.99:
                new_sentences.append(sentence) 
    return new_sentences


def process_data(text,token): 
    text = text.replace('.', '.<eos>')
    text = text.replace('?', '?<eos>')
    text = text.replace('!', '!<eos>') 
    sentences = text.split('<eos>')  
    sentences =  [s for s in sentences if len(s) > 5]
    new_sen = run_sentimental_model(sentences,token) 
    if len(new_sen) == 0:
        return new_sen
    return new_sen



def write_to_json(classificationDict,token):
    kek = copy.deepcopy(classificationDict)

    file_path = "data.json"

    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    if token not in data:
        data[token] = {}

    for k,v in kek.items():
        if k not in data[token]:
            data[token][k] = list(set(v))
        else:
            data[token][k].extend([s for s in v]) 

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
        



def process_data2(sentences,token): 
    REPORT = {}
    for s in sentences:
        kek = classifier2(s, candidate_labels=LABELS,device=device)    
        high_labeled = [kek['labels'][i] for i in  range(len(kek['labels'])) if float(kek['scores'][i]) > 0.1 ]  
        for label in high_labeled:
            if label in REPORT:
                REPORT[label].append(s)
            else:
                REPORT[label] = [s] 
    write_to_json(REPORT, token)



def gpt_generate(topic): 

    openai.api_key =  "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"   
    # Specify the path to the JSON file
    file_path = "data.json"

    # Open and read the JSON file
    with open(file_path, "r") as json_file:
        data = json.load(json_file) 

    prompt = f"Given the data { str(data[topic])}. Write a short summary of the text in  each cateogry."

    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

    return completion.choices[0].message.content


@app.route('/classify',  methods=['POST'])
def run():
    data = request.get_json()
    token = data["company"]  

    
    res = google_search(token)  

    if res == None:
        result = {'message': "no results on google" }
        return jsonify(result), 400 

    links  = extract_links(res) 

    
    parse_urls_if_exist(links,token)  


    with open(f'example.txt', 'r', encoding='utf-8') as file:
            file_content = file.read()  

   
    sentences = process_data(file_content,token)   


    process_data2(sentences,token)

    # with open('example.txt', 'a',encoding='utf-8') as file:
    #     file.write("")   

    res = gpt_generate(token)
    result = {'info': res} 
    return jsonify(result), 200  


app.run(port=8080)
