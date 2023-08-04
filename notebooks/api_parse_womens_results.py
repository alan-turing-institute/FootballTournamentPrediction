import json

import pandas as pd

with open("results.json") as f:
    data = json.load(f)

results = [
    {
        "date": f["fixture"]["date"],
        "home_team": f["teams"]["home"]["name"],
        "away_team": f["teams"]["away"]["name"],
        "home_score": f["goals"]["home"],
        "away_score": f["goals"]["away"],
        "tournament": "FIFA World Cup",
        "city": f["fixture"]["venue"]["city"],
        "country": None,
        "neutral": None,
    }
    for f in data["response"]
]

hosts = ["Australia W", "New Zealand W"]
for result in results:
    if result["home_team"] in hosts or result["away_team"] in hosts:
        result["neutral"] = "FALSE"
    else:
        result["neutral"] = "TRUE"

cities = {
    "Auckland": "New Zealand",
    "Sydney": "Australia",
    "Melbourne": "Australia",
    "Dunedin": "New Zealand",
    "Wellington": "New Zealand",
    "Hamilton": "New Zealand",
    "Brisbane": "Australia",
    "Perth": "Australia",
    "Adelaide": "Australia",
}
for result in results:
    result["country"] = cities[result["city"]]

df = pd.DataFrame(results)
df["home_team"] = df["home_team"].str.replace(" W", "")
df["away_team"] = df["away_team"].str.replace(" W", "")
df["date"] = pd.to_datetime(df["date"]).dt.date

df.to_csv("results.csv", index=False)
