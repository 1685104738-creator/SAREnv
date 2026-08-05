import requests
import time

STATUS_URL = "https://overpass-api.de/api/status"
INTERPRETER_URL = "https://overpass-api.de/api/interpreter"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    "Referer": "https://overpass-api.de/",
    "Accept": "text/html,application/json,text/plain,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}

print("Testing status endpoint with headers...")
start = time.time()

response = requests.get(
    STATUS_URL,
    headers=headers,
    timeout=30,
)

print("Status:", response.status_code)
print("Elapsed:", time.time() - start)
print(response.text[:1000])


print("\nTesting tiny Overpass query with headers...")

query = """
[out:json][timeout:25];
node(around:100,55.145921,10.289470);
out 1;
"""

start = time.time()

response = requests.post(
    INTERPRETER_URL,
    data={"data": query},
    headers=headers,
    timeout=30,
)

print("Status:", response.status_code)
print("Elapsed:", time.time() - start)
print(response.text[:1000])