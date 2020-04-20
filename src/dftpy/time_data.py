import time

class TimeObj(object):
    """
    """

    def __init__(self, **kwargs):
        self.reset(**kwargs)

    def reset(self, **kwargs):
        self.labels = []
        self.tic = {}
        self.toc = {}
        self.cost = {}
        self.number = {}

    def Begin(self, label):
        if label in self.tic:
            self.number[label] += 1
        else:
            self.labels.append(label)
            self.number[label] = 1
            self.cost[label] = 0.0

        self.tic[label] = time.time()

    def Time(self, label):
        if label not in self.tic:
            print(' !!! ERROR : You should add "Begin" before this')
        else:
            t = time.time() - self.tic[label]
        return t

    def End(self, label):
        if label not in self.tic:
            print(' !!! ERROR : You should add "Begin" before this')
        else:
            self.toc[label] = time.time()
            t = time.time() - self.tic[label]
            self.cost[label] += t
        return t

    def output(self, config):
        print(format("Time information", "-^80"))
        print("{:28s}{:24s}{:20s}".format("Label", "Cost(s)", "Number"))
        if config["OUTPUT"]["time"]:
            for key, cost in sorted(self.cost.items(), key=lambda d: d[1]):
                print("{:28s}{:<24.4f}{:<20d}".format(key, cost, self.number[key]))
        else:
            key = "TOTAL"
            print("{:28s}{:<24.4f}{:<20d}".format(key, self.cost[key], self.number[key]))


TimeData = TimeObj()
