import collections
import timeit

class Profiler:
    """
    Profiler
    This class provides a simple profiling mechanism for measuring the execution time of different sections of code. It records the time taken for each named section of code and calculates the mean and variance of the recorded times.

    Methods:
        __init__(): Initialize the Profiler object with default settings.
        reset(): Reset the profiler data and start tracking time from the current point.
        time(name): Record the execution time for a specific section of code with the given name.
        means(): Get the dictionary of mean times for each recorded section.
        vars(): Get the dictionary of variances of the recorded times for each section.
        stds(): Get the dictionary of standard deviations of the recorded times for each section.
        total(): Calculate the total time taken for all recorded sections.
        summary(prefix=""): Generate a summary of the profiling data with mean time, standard deviation, and percentage of total time for each recorded section.

    Usage::
        The Profiler object can be used to measure the execution time of different parts of a code by calling the `time` method with a descriptive name for each section. After profiling, the `summary` method can be used to print a summary of the profiling data.

    Example::
        profiler = Profiler()
        for i in range(10):
            # Code segment to be profiled
            profiler.time("Code Segment %d" % i)

        print(profiler.summary("Profiling Results:"))
    """

    def __init__(self):
        self._means = collections.defaultdict(int)
        self._vars = collections.defaultdict(int)
        self._counts = collections.defaultdict(int)
        self.reset()

    def reset(self):
        self.last_time = timeit.default_timer()

    def time(self, name):
        now = timeit.default_timer()
        x = now - self.last_time
        self.last_time = now

        n = self._counts[name]

        mean = self._means[name] + (x - self._means[name]) / (n + 1)
        var = (n * self._vars[name] + n * (self._means[name] - mean) ** 2 + (x - mean) ** 2) / (n + 1)

        self._means[name] = mean
        self._vars[name] = var
        self._counts[name] += 1

    def means(self):
        return self._means

    def vars(self):
        return self._vars

    def stds(self):
        return {k: v ** 0.5 for k, v in self._vars.items()}

    def total(self):
        return sum(self.means().values())

    def summary(self, prefix=""):
        means = self.means()
        stds = self.stds()
        total = sum(means.values())

        result = prefix
        for k in sorted(means, key=means.get, reverse=True):
            result += f"\n   -> %s: %.3fms +- %.3fms (%.2f%%) " % (
                k,
                1000 * means[k],
                1000 * stds[k],
                100 * means[k] / total,
            )
        result += "\nTotal: %.3fms" % (1000 * total)
        return result
