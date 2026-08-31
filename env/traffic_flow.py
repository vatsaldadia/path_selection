class TrafficFlow:

    def __init__(self, flow: dict):

        self.ru = flow["ru"]
        self.upf_location = flow["upf_location"]
        self.bandwidth = flow["bandwidth"]
        self.latency = flow["latency"]
        self.direction = flow["direction"]
        self.slice = flow["slice"]
        self.service = flow["service"]
        self.duration = flow["duration"]

        self.chosen_path: list[str] = []
        self.chosen_datacenters: list[str] = []
        self.power_consumed: float = 0.0
        self._granted: bool = False

    def grant(self):
        self._granted = True

    def is_granted(self):
        return self._granted

    def __repr__(self):
        return f"{self.ru=}, {self.upf_location=}, {self.bandwidth=}, {self.latency=}, {self.direction=}, {self.duration=}"

    def __lt__(self, other):            # tie-breaker for heap
        return self.bandwidth < other.bandwidth