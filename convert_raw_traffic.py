import json
import csv

with open('raw_traffic.json') as f:
    data = json.load(f)

NODE_TYPE_ATTRIBUTES = {
    'RU': {
        "processing_time": 10,
    },
    'DU': {
        "processing_time": 6,
        "resources": {
            "cpu": 9,
            "memory": 5,
            "storage": 1
        },
        "power_params": {
            "idle": 100,
            "max": 350,
            "alpha": 1.3
        }
    },
    'CU': {
        "processing_time": 16,
        "resources": {
            "cpu": 11,
            "memory": 15,
            "storage": 2
        },
        "power_params": {
            "idle": 120,
            "max": 400,
            "alpha": 1.3
        }
    },
    "UPF_TN": {
        "processing_time": 4,
        "resources": {
            "cpu": 5,
            "memory": 4,
            "storage": 7
        },
        "power_params": {
            "idle": 90,
            "max": 300,
            "alpha": 1.3
        }
    },
    "UPF_CN": {
        "processing_time": 6,
        "resources": {
            "cpu": 5,
            "memory": 4,
            "storage": 7
        },
        "power_params": {
            "idle": 90,
            "max": 300,
            "alpha": 1.3
        }
    },
    'switch': {
        "power_params": {
            "idle": 20,
            "bit": 0.001,
        }
    }
}

service_mapper = {
    "Video streaming": "VS",
    "Cloud Gaming": "CG",
    "VoIP": "VOIP",
    "Massive IoT": "MIOT",
    "Industry 4.0": "I4.0",
    "Augmented reality": "AR"
}

location_mapper = {
        "VS": "CN",
        "CG": "CN",
        "VOIP": "CN",
        "MIOT": "TN",
        "I4.0": "TN",
        "AR": "TN"
}

latency_mapper = {
    "VS": 100,
    "CG": 80,
    "VOIP": 100,
    "MIOT": 10,
    "I4.0": 15,
    "AR": 20
}

max_bw_mapper = {}
for service, location in location_mapper.items():
    max_bw_mapper[service] = (latency_mapper[service] - 2) / (sum(NODE_TYPE_ATTRIBUTES[func]["processing_time"]
           for func in ["RU", "DU", "CU", f"UPF_{location}"]) * 0.0001)           # keep 2s for propagation
print(max_bw_mapper)

# def get_processing_delay(service, bandwidth):
#     return sum(NODE_TYPE_ATTRIBUTES[func]["processing_time"]
#                for func in ["RU", "DU", "CU", f"UPF_{location_mapper[service]}"]) * bandwidth * 0.0001
#
# data_to_write = []
# for hour in data:
#     for flow in data[hour]:
#         flow["hour"] = hour
#         direction = "UL" if flow["src"].startswith("RU") else "DL"
#         flow["ru"] = flow["src"] if direction == "UL" else flow["dst"]
#         flow["ru"] = "_".join(flow["ru"].split("_")[:-1])
#         flow["direction"] = direction
#         del flow["src"]
#         del flow["dst"]
#         flow["service"] = service_mapper[flow["service"]]
#         if flow["total_bandwidth"] > max_bw_mapper[flow["service"]]:
#             flow["total_bandwidth"] = max_bw_mapper[flow["service"]]
#         flow["bandwidth"] = round(flow["total_bandwidth"], 4)
#         # flow["processing_delay"] = round(get_processing_delay(flow["service"], flow["bandwidth"]), 4)
#         flow["slice"] = flow["class_name"]
#         del flow["total_bandwidth"]
#         del flow["num_user_per_request"]
#         del flow["functions"]
#         data_to_write.append(flow)
#
#
# with open(f"month_0.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["hour", "direction", "ru", "bandwidth", "service", "delay", "slice"])
#     for d in data_to_write:
#         writer.writerow([d["hour"], d["direction"], d["ru"], d["bandwidth"], d["service"], d["delay"], d["slice"]])