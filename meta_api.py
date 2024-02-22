import requests
import pandas as pd

response_data = {"id": "66badd9d-66bb-4227-9781-daff02a5383a"}
session_id = response_data['id']

headers = {'X-Metabase-Session': session_id}

metabase_url = "https://metabase.medelse.com/api/card/893/query/csv"

variable_args = {
    "announcement_id": "96650",
}

with requests.Session() as session:
    session.headers.update(headers)
    payload = {
        "parameters": variable_args
    }

    response = session.post(metabase_url, json=payload)
    response.raise_for_status()

    csv_file_path = "single_query.csv"

    with open(csv_file_path, "wb") as csv_file:
        csv_file.write(response.content)

    print(f"CSV result saved to {csv_file_path}")