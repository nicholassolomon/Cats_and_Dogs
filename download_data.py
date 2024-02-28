import requests
import os
import zipfile
from pathlib import Path

def download_zip_data(url, path_to_data):
  """
  Download zipped contents and unzip
  """
  path = Path(url)
  print(f"full name: {path.name}")
  print(f"name only: {path.stem}")
  data_path = Path(path_to_data)
  print(str(data_path))

  if data_path.is_dir():
    print(f"path_to_data exists")
  else:
    print(f"path doesn't exist")
    data_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / path.name, "wb") as f:
        request = requests.get(url)
        print(f"Downloading {path.stem}")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / path.name, "r") as zip_ref:
        print(f"Unzipping {path.stem}")
        zip_ref.extractall(data_path)

    os.remove(data_path / path.name)

