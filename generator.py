import pickle
import csv
from collections import defaultdict
import torch as th

class Generator:
    def __init__(self, pkl_file="./init_modified.pkl"):
        self.traffic = defaultdict(list)
        self.traffic_start_hour = 0
        self.env_last_hour = -1
        self.csv_traffics = []
        self.total_count = 0

        with open(pkl_file, "rb") as f:
            self.network_data = pickle.load(f)

    def load_traffic(self, csv_file):
        self.csv_traffics.append(csv_file)

        location_mapper = {
            "CG": "CN",
            "AR": "TN",
            "VOIP": "CN",
            "VS": "CN",
            "MIOT": "TN",
            "I4.0": "TN"
        }

        reader = csv.DictReader(open(csv_file))
        for flow in reader:
            hour = (int(flow["hour"]))
            self.traffic[self.traffic_start_hour + hour].append({
                "hour": hour,
                "direction": flow["direction"],
                "service": flow["service"],
                "slice": flow["slice"],
                "ru": flow["ru"],
                "upf_location": location_mapper[flow["service"]],
                "bandwidth": (float(flow["bandwidth"])),
                "latency": (float(flow["delay"])),
                "duration": (int(flow["duration"])),
            })
            self.total_count += 1
        self.traffic_start_hour = max(self.traffic.keys()) + 1

    def get_hour_traffic(self, hour):
        return self.traffic[hour].copy()

    def can_do_new_task(self):
        if self.traffic_start_hour == 0:
            raise ValueError("No traffic loaded, stopping training. Please load traffic first.")
        return (self.env_last_hour + 1) < self.traffic_start_hour

    def get_network_data(self):
        return self.network_data

    def update_env_hour(self, hour):
        self.env_last_hour = hour

    def get_csv_traffics(self):
        return self.csv_traffics

    def get_total_traffic_count(self):
        return self.total_count

    def get_max_bw(self):
        service_max_bw = defaultdict(int)
        for hour, flows in self.traffic.items():
            for f in flows:
                if f["bandwidth"] > service_max_bw[f["service"]]:
                    service_max_bw[f["service"]] = f["bandwidth"]
        return service_max_bw

if __name__ == "__main__":
    g = Generator()