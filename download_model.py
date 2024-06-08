import requests
import os
from dotenv import load_dotenv

load_dotenv()
model_link = os.getenv('MODEL_URI')

def download_model():
    try:
        model_path = './model/pneu-model.keras'
        if not os.path.exists(model_path):
            print(f"Starting model download", flush=True)
            response = requests.get(model_link)
            response.raise_for_status() 
            print(f"Response status code: {response.status_code}", flush=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)  
            with open(model_path, 'wb') as file:
                print('Downloading...', flush=True)
                file.write(response.content)
            message = f"Model downloaded successfully: {model_path}"
        else:
            message = f"Model exists: {model_path}"
            return message
    except requests.exceptions.RequestException as e:
        print(f"Failed to download Model. Error: {e}", flush=True)
