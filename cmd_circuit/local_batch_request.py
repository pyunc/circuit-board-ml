import requests

local_path_folder = '/home/paulo/Documents/circuit-board-ml/webapp_fastapi/organized_data'

headers = headers = {'Content-Type': 'application/parquet'}

res = requests.post(url='http://localhost:8081/invocations', data=local_path_folder, headers=headers)

print(res.json())
