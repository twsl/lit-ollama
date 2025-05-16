import requests

response = requests.post("http://127.0.0.1:11434/predict", json={"input": 4.0}, timeout=5000)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
