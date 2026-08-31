import json
import csv

traffic_name = "month_0"

with open(f"{traffic_name}.json") as f:
    data = json.load(f)

data_to_write = []
for hour in data.keys():
    for f in data[hour]:
        f["hour"] = hour
        data_to_write.append(f)

with open(f"{traffic_name}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["hour", "direction", "ru", "bandwidth", "service", "delay", "slice"])
    for d in data_to_write:
        writer.writerow([d["hour"], d["direction"], d["ru"], d["bandwidth"], d["service"], d["delay"], d["slice"]])