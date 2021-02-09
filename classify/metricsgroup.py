class MetricsGroup:
        
    def __init__(self, metrics_dict):
        self.metrics = metrics_dict
        
    def update(self, output):
        for name, metric in self.metrics.items():
            metric.update(output)
            
    def compute(self):
        output = {}
        for name, metric in self.metrics.items():
            output[name] = metric.compute()
        return output
