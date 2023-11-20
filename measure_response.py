import requests
import time
import numpy as np

from tqdm import tqdm

if __name__ == "__main__":

    application = {
        "session_id": "9055446045651783499.1640648526.1640648526",
        "client_id": "2108385331.1640648523",
        "visit_date": "2021-12-28",
        "visit_time": "02:42:06",
        "visit_number": 1,
        "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
        "utm_medium": "banner",
        "utm_campaign": "LEoPHuyFvzoNfnzGgfcd",
        "utm_adcontent": "vCIpmpaGBnIQhyYNkXqp",
        "utm_keyword": "puhZPIYqKXeFPaUviSjo",
        "device_category": "mobile",
        "device_os": "Android",
        "device_brand": "Huawei",
        "device_model": None,
        "device_screen_resolution": "360x720",
        "device_browser": "Chrome",
        "geo_country": "Russia",
        "geo_city": "Krasnoyarsk"
    }

    url = "http://127.0.0.1:73/predict/"

    all_times = []
    # Count for 1000
    for i in tqdm(range(1000)):
        t0 = time.time()
        # A request
        resp = requests.post(url, json=application)
        t1 = time.time()
        # Measure the respond time
        time_taken = t1 - t0
        all_times.append(time_taken)

    average_time = np.mean(all_times)
    print(f"Average response time: {average_time:.4f} seconds")
