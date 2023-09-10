
import requests
from bs4 import BeautifulSoup 
from flask import Flask, jsonify, request 
import threading

app = Flask(__name__)   

SENTSCORE_URL = "http://localhost:8001/ss"


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

            # data = {
            #   "text_page": linked_page_text,
            #   "company": token} 
            
            # json_data = json.dumps(data)
            # headers = {"Content-Type": "application/json"}
            # res = requests.post(SENTSCORE_URL, data=json_data, headers=headers) 
            # if res.status_code == 200:
            #     print(res.text)
            # else:
            #     print(f"POST request failed with status code: {res.status_code}") 



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
    files = {'file': ('example.txt', open('example.txt', 'rb'))} 
    data = {'company': company} 

    response = requests.post(SENTSCORE_URL, files=files,data=data)

    with open('example.txt', 'w',encoding='utf-8') as file: 
        file.write(" ")  

    return  {'message': response.text}
    
 


@app.route('/scrape')
def scrape():
    data = request.get_json()
    token = data["company"]
    res = google_search(token)  

    if res == None:
        result = {'message': "no results on google" }
        return jsonify(result), 400 

    links  = extract_links(res) 

    if len(links) == 0:
        result = {'message': "company's links not found" }
        return  jsonify(result), 400  
    
    return parse_urls_if_exist(links,token) 

    

app.run(port=8000)