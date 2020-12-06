import time
from dftpy.mpi import sprint

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

    def output(self, config=None, sort = 0, comm = None):
        """
        sort : Label(0), Cost(1), Number(2), Avg(3)
        """
        column = {
                'Label'  : 0,
                'Cost'   : 1,
                'Number' : 2,
                'Avg'    : 3,
                }
        if sort in column :
            idx = column[sort]
        elif isinstance(sort, (int, float)):
            idx = int(sort)
            if idx < 0 or idx > 3 :
                idx = 0
        else :
            idx = 0
        sprint(format("Time information", "-^80"), comm = comm)
        sprint("{:28s}{:24s}{:16s}{:24s}".format("Label", "Cost(s)", "Number", "Avg. Cost(s)"), comm = comm)
        lprint = False
        if config :
            if isinstance(config, dict) and not config["OUTPUT"]["time"]:
                lprint = False
            else :
                lprint = True
        if lprint :
            info = []
            for key, cost in self.cost.items():
                if key == 'TOTAL' : continue
                item = [key, cost, self.number[key], cost/self.number[key]]
                info.append(item)
            for item in sorted(info, key=lambda d: d[idx]):
                sprint("{:28s}{:<24.4f}{:<16d}{:<24.4f}".format(*item), comm = comm)
        key = "TOTAL"
        sprint("{:28s}{:<24.4f}{:<16d}{:<24.4f}".format(key, self.cost[key], self.number[key], self.cost[key]/self.number[key]), comm = comm)
        # print(sorted(self.toc.keys()))


TimeData = TimeObj()
