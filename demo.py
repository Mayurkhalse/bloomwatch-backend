import requests
import json

url = "http://localhost:8000/store-data"

import json

# Open and load the JSON file
with open("bloom_full_history_forecast.json", "r") as f:
    data = json.load(f)


# Send POST request
response = requests.post(url, json=data)

# Print response
print(response.status_code)
print(response.json())
