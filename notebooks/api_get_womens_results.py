import json

import requests

headers = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": "<YOUR_API_KEY>",
}

response = requests.get(
    "https://v3.football.api-sports.io/fixtures",
    headers=headers,
    params={
        "league": 8,
        "status": "FT",
        "season": 2023,
    },
)

with open("results.json", "w") as f:
    json.dump(response.json(), f)
