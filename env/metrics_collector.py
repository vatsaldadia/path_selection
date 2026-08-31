from env.traffic_flow import TrafficFlow


class MetricsCollector:

    # noinspection unsupported-operator,bad-index
    class HourlyMetrics:
        def __init__(self):

            self.result_value: dict[str, int | float | dict[str, dict[str, int | float]]] = {
                "encountered": 0,
                "granted": 0,
                "power_consumed": 0.0,
                "GoS": 0.0,
                "processing_time_error": 0,
                "datacenter_cpu_dead_end": 0,
                "latency_dead_end": 0,
                "bandwidth_dead_end": 0
            }

            self.location_results: dict[str, dict[str, int | float]] = {
                "TN": {},
                "CN": {}
            }
            for i in self.location_results:
                self.location_results[i] = self.result_value.copy()

            self.slice_results = {
                "eMBB": {},
                "uRLLC": {},
                "mMTC": {}
            }
            for i in self.slice_results:
                self.slice_results[i] = self.result_value.copy()

            self.service_results = {
                "CG": {},
                "AR": {},
                "VOIP": {},
                "VS": {},
                "MIOT": {},
                "I4.0": {}
            }
            for i in self.service_results:
                self.service_results[i] = self.result_value.copy()

        def _update_metric_by_value(self, metric: str, flow: TrafficFlow, value: int | float):
            self.result_value[metric] += value
            self.slice_results[flow.slice][metric] += value
            self.service_results[flow.service][metric] += value

        def _update_GoS(self):
            self.result_value["GoS"] = self.result_value["granted"] / self.result_value["encountered"] if self.result_value["encountered"] > 0 else 0
            for location_name, location_result in self.location_results.items():
                location_result["GoS"] = location_result["granted"] / location_result["encountered"] if location_result["encountered"] > 0 else 0
            for slice_name, slice_result in self.slice_results.items():
                slice_result["GoS"] = slice_result["granted"] / slice_result["encountered"] if slice_result["encountered"] > 0 else 0
            for service_name, service_result in self.service_results.items():
                service_result["GoS"] = service_result["granted"] / service_result["encountered"] if service_result["encountered"] > 0 else 0

        def update_metrics(self, flow: TrafficFlow, description: str):
            val = 1 if description != "power_consumed" else flow.power_consumed
            self._update_metric_by_value(description, flow, val)
            if flow.is_granted():
                self._update_GoS()  # TODO: optimize so invoke is only after hour end, not every granted flow

        def return_metrics(self):
            self.result_value["location_results"] = self.location_results
            self.result_value["service_results"] = self.service_results
            self.result_value["slice_results"] = self.slice_results
            return self.result_value

    def __init__(self):
        self.hourly_results = {}
        self.cpu_metrics = {}

    def update_hourly_metrics(self, hour: int, flow: TrafficFlow, description: str):
        if hour not in self.hourly_results:
            # print(hour)
            self.hourly_results[hour] = self.HourlyMetrics()
        self.hourly_results[hour].update_metrics(flow, description)

    def get_total_metric_value(self, metric: str):
        if metric == "GoS":
            return self.get_total_metric_value("granted") / self.get_total_metric_value("encountered") if self.get_total_metric_value("encountered") > 0 else 0
        total = 0
        for hourly_result in list(self.hourly_results.values()):
            total += hourly_result.result_value[metric]
        return total

    def update_cpu_metrics(self, hour: int, node_data):
        if hour == -1:
            return
        self.cpu_metrics[hour] = {}
        for node, data in node_data.items():
            self.cpu_metrics[hour][node] = {
                "utilization": 1 - (data["available"]["cpu"] / data["capacity"]["cpu"]),
                "available": data["available"]["cpu"],
                "capacity": data["capacity"]["cpu"]
            }


    def return_hourly_results(self):
        return_dict = {}
        for hour, hourly_result in self.hourly_results.items():
            return_dict[hour] = hourly_result.return_metrics()
        return return_dict

    def return_cpu_metrics(self):
        return self.cpu_metrics

    def return_info(self):
        return {
            "encountered": self.get_total_metric_value("encountered"),
            "granted": self.get_total_metric_value("granted"),
            "GoS": self.get_total_metric_value("GoS"),
            "datacenter_cpu_dead_end": self.get_total_metric_value("datacenter_cpu_dead_end"),
            "bandwidth_dead_end": self.get_total_metric_value("bandwidth_dead_end"),
            "latency_dead_end": self.get_total_metric_value("latency_dead_end"),
            "processing_time_error": self.get_total_metric_value("processing_time_error"),
        }