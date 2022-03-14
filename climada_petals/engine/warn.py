
class Warn:

    kernel_sizes = [1, 3, 7, 15,
                    1, 3, 7, 15,
                    2, 3, 7, 15,
                    1, 4, 8, 15]

    def __init__(self, thresholds, data, ):
        self.threshold = thresholds
        self.data = data

