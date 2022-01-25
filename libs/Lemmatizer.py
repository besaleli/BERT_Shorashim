import requests

class Lemmatizer:
    
    def divide_list(l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]
        
    def getData(sent):
        localhost_yap = "http://localhost:8000/yap/heb/joint"
        headers = {'content-type': 'application/json'}
        
        text = ""
        
        punct = ["'", '"', u'\xa0']
        for i in range(len(sent) - 1):
            if sent[i] not in punct:
                text += sent[i]
            else:
                text += ' '
        
        data = '{{"text": "{}  "}}'.format(text).encode('utf-8')
        response = requests.get(url=localhost_yap, data=data, headers=headers)
        response_json = response.json()
        return response_json['md_lattice']
    