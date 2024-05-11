class RunningMean():
    def __init__(self):
        """
            This class is used to calculate the running mean (moving average of a sequence)
        """
        self.mean = 0   # Current mean
        self.n = 0      # Number of samples

    def update(self, x, n):
        self.mean = (self.mean * self.n + x * n) / (self.n + n)
        self.n += n

    def reset(self):
        self.mean = 0
        self.n = 0

    def __call__(self):
        return self.mean
