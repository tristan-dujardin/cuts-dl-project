import os
import zipfile
from urllib import request

url = "https://github.com/KrishnaswamyLab/CUTS/raw/main/data/retina.zip"
data_path = "data/"
zip_path = "data/retina.zip"

os.makedirs(os.path.dirname(zip_path), exist_ok=True)

request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_path)

print(f"Files extracted to: {data_path}")