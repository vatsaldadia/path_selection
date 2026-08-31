class DriftDetector:
    def __init__(self, threshold=24):
        self.threshold = threshold

    def is_drift(self, hour):
        return hour % self.threshold == 0 if hour > 0 else False