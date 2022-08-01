import requests
import os
url = "http://127.0.0.1:5000/inference"
# url = "http://10.225.20.177:5000/inference"
headers = {"accept": "application/json"}
root = "/home/sangdaen/dsa_demo/data"
image_list = [os.path.join(root, 'inference_c_new', i) for i in os.listdir(os.path.join(root, 'inference_c_new'))][:128]

files = {}
for i,v in enumerate(image_list):
    key = f'img{i}'
    files[key] = open(v, "rb")

response = requests.post(url, headers=headers, files=files)
breakpoint()

print(response.json())
