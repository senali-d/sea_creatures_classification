import requests
import json
# print(json.dumps(data))  # This should work without errors

url = 'http://localhost:8080/2024-11-07/functions/function/invocations'

data = {'url': 'https://raw.githubusercontent.com/senali-d/sea_creatures_classification/refs/heads/main/data/Clams/10004815975_14842bfd9d_o.jpg'}

result = requests.post(url, json=data).json()
print(result)