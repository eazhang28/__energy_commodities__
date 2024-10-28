import requests

API_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/?api_key=xZfc9smFchx7pxAglwzNwLhXJTbtaYVGTijcoab1&data[0]=price&data[1]=revenue"

def fetch_data():
    response = requests.get(API_URL)
    data = response.json()
    print(data["response"]["data"][0])

if __name__ == "__main__":
    fetch_data()