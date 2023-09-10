from flask import Flask, jsonify, request 

from transformers import pipeline
import torch  
import copy
import json


app = Flask(__name__)   


classifier = pipeline("zero-shot-classification")  
LABELS = ["historic", "financial" ,  "products", "services", "general activity", "business model", "areas of expertise",  "news"] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)



def process_data(sentences,token):
    REPORT = {}
    for s in sentences:
        kek = classifier(s, candidate_labels=LABELS,device=device)    
        high_labeled = [kek['labels'][i] for i in  range(len(kek['labels'])) if float(kek['scores'][i]) > 0.2 ]  
        for label in high_labeled:
            if label in REPORT:
                REPORT[label].append(s)
            else:
                REPORT[label] = [s] 
    write_to_json(REPORT, token)



def write_to_json(classificationDict,token):
    kek = copy.deepcopy(classificationDict)

    file_path = "data.json"

    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    if token not in data:
        data[token] = {}

    for k,v in kek.items():
        if k not in data[token]:
            data[token][k] = v
        else:
            data[token][k].extend([s for s in v]) 

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)




@app.route('/classify',  methods=['POST'])
def run():
    data = request.get_json()
    sentences = data["sentences"] 
    token = data["company"] 
    
    #classification_wf = threading.Thread(target=process_data, args=(sentences,token))
    #classification_wf.start() 
    process_data(sentences, token)

    result = {'message': "ok" }
    return jsonify(result), 200  


    

app.run(port=8002)