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

POWER_VALUES = {
    "alpha": {"a1": 0.20, "a2": 0.25, "a3": 0.25,
              "a4": 0.30, "a5": 0.5, "a6": 0.20,
              "a7": 0.05},
    "freq_Hz": 2.5e9,
    "freq_idle_Hz": 2.0e9,
    "w_W_per_Hz": 1e-10,
}

REWARD_SYSTEM = {
    "success": +1.0,  # Reward for successfully completing the entire flow
    "scaling_alpha": 0.1,  # Reward for scaling the ln(min_avail_cpu)
    "failure": 0.0,  # Penalty for failing
}