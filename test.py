import requests
import json
# print(json.dumps(data))  # This should work without errors

url = 'http://localhost:8080/2024-11-07/functions/function/invocations'

data = {'url': './data/Clams/10711395_a16c4c2901_o.jpg'}

result = requests.post(url, json=data).json()
print(result)