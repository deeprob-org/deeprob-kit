class RunningAverageMetric:
    def __init__(self):
        """
        Initialize a running average metric object.
        """
        self.__samples_counter = 0
        self.__metric_accumulator = 0.0

    def __call__(self, metric: float, num_samples: int):
        """
        Accumulate a metric value.

        :param metric: The metric value.
        :param num_samples: The number of samples from which the metric is estimated..
        :raises ValueError: If the number of samples is not positive.
        """
        if num_samples <= 0:
            raise ValueError("The number of samples must be positive")
        self.__samples_counter += num_samples
        self.__metric_accumulator += metric * num_samples

    def reset(self):
        """
        Reset the running average metric accumulator.
        """
        self.__samples_counter = 0
        self.__metric_accumulator = 0.0

    def average(self) -> float:
        """
        Get the metric average.

        :return: The metric average.
        """
        return self.__metric_accumulator / self.__samples_counter
