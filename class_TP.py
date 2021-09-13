
from __future__ import barry_as_FLUFL
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import scipy.sparse as spa
import cvxpy as cp
from math import radians, cos, sin, asin, sqrt


class TP:

    def __init__(self, data, taxis=3, taxi_lon=-73.9772, taxi_lat=40.7527, mph=20, driving_cost=.6, taxi_cost=0, flex=20, waiting_cost=0.1):

        n = len(data.index)
        source = n
        sink = n+1
        num_nodes = n+2
        year = data.loc[0, 'pickup_datetime'].year
        month = data.loc[0, 'pickup_datetime'].month
        day = data.loc[0, 'pickup_datetime'].day
        midnight = datetime.datetime(year, month, day, 0, 0)
        flex = timedelta(minutes=flex)

        # =1 if connection (i,j) is possible, =0 if impossible
        possible = np.zeros((num_nodes, num_nodes))
        # Net profit from accepting passenger j (based on preceding passenger i)
        profits = np.zeros((num_nodes, num_nodes))
        # Note: =0 if connection (i,j) is impossible
        Tmatrix = np.zeros((num_nodes, num_nodes))

        for i in range(n):
            possible[i, sink] = 1
            Tmatrix[i, sink] = -9999999
#             dist_between = self.haversine(data.loc[i, 'dropoff_longitude'], data.loc[i, 'dropoff_latitude'], taxi_lon, taxi_lat)
#             profits[i, sink] = -driving_cost * dist_between
            for j in range(n):
                if (i == j):
                    continue
                if (possible[j, i] == 1):
                    continue
                dist_between = self.haversine(data.loc[i, 'dropoff_longitude'], data.loc[i, 'dropoff_latitude'],
                                              data.loc[j, 'pickup_longitude'], data.loc[j, 'pickup_latitude'])
                time_between = timedelta(hours=dist_between / mph)
                if (data.loc[i, 'dropoff_datetime'] + time_between > data.loc[j, 'pickup_datetime'] + flex):
                    continue
                else:
                    possible[i, j] = 1
                    Tmatrix[i, j] = self.to_minutes(
                        data.loc[i, 'dropoff_datetime'] - data.loc[i, 'pickup_datetime'] + time_between)
                    profits[i, j] = data.loc[j, 'total_amount'] - \
                        driving_cost * (dist_between +
                                        data.loc[j, 'trip_distance'])

        for j in range(n):
            # Taxis (Source to First Customers)
            dist_between = self.haversine(
                taxi_lon, taxi_lat, data.loc[j, 'pickup_longitude'], data.loc[j, 'pickup_latitude'])
            time_between = timedelta(hours=dist_between / mph)
            if (midnight + time_between > data.loc[j, 'pickup_datetime'] + flex):
                continue
            else:
                possible[source, j] = 1
                profits[source, j] = data.loc[j, 'total_amount'] - driving_cost * \
                    (dist_between + data.loc[j, 'trip_distance']) - taxi_cost
                Tmatrix[source, j] = self.to_minutes(time_between)

        arcs = {}
        a = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if (possible[i, j] == 1):
                    arcs[a] = (i, j)
                    a += 1
                if (possible[j, i] == 1):
                    arcs[a] = (j, i)
                    a += 1

        self.possible = possible
        self.data = data
        self.n = n
        self.source = source
        self.sink = sink
        self.num_nodes = num_nodes
        self.num_taxis = taxis
        self.taxi_lon = taxi_lon
        self.taxi_lat = taxi_lat
        self.mph = mph
        self.midnight = midnight
        self.profits = profits
        self.arcs = arcs
        self.num_arcs = len(arcs)
        self.waiting_cost = waiting_cost
        self.Tmatrix = Tmatrix

    # Set-Up Functions for Optimization Problems

    def flow_vars(self):
        A_in = spa.dok_matrix((self.num_nodes, self.num_arcs))
        A_out = spa.dok_matrix((self.num_nodes, self.num_arcs))
        p = np.zeros(self.num_arcs)
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            A_in[i, a] = 1
            A_out[j, a] = 1
            p[a] = self.profits[i, j]

        e = np.zeros(self.num_nodes)
        e[self.source] = self.num_taxis
        e[self.sink] = self.num_taxis

        return A_in, A_out, e, p

    def time_vars(self, time_window):
        B = spa.dok_matrix((self.num_arcs, self.num_arcs))
        d = np.zeros(self.num_arcs)

        t_min, t_max = self.time_cons(time_window)
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            B[a, a] = t_max[i] - t_min[j] + self.T(i, j)
            d[a] = t_max[i] - t_min[j]
        return B, d

    def time_cons(self, time_window):
        t_min = []
        t_max = []
        time_window = timedelta(minutes=time_window)
        for i in range(self.n):
            t_min.append(self.to_minutes(
                self.data.loc[i, 'pickup_datetime'] - self.midnight))
            t_max.append(self.to_minutes(
                self.data.loc[i, 'pickup_datetime'] - self.midnight + time_window))
        for k in range(self.n, self.num_nodes):
            t_min.append(0)
            t_max.append(0)
#         hours_24 = timedelta(hours=24)
#         t_min.append(self.to_minutes(self.midnight + hours_24))
#         t_max.append(self.to_minutes(self.midnight + hours_24))
        return t_min, t_max

    def T(self, i, j):
        if (i == j):
            return 0
        # From customer to sink
        if (j == self.sink):
            return -9999999
        # From taxi to customer
        if (i == self.source):

            dist_between = self.haversine(
                self.taxi_lon, self.taxi_lat, self.data.loc[j, 'pickup_longitude'], self.data.loc[j, 'pickup_latitude'])
            output = timedelta(hours=dist_between / self.mph)
        # From customer to customer
        else:

            dist_between = self.haversine(self.data.loc[i, 'dropoff_longitude'], self.data.loc[i, 'dropoff_latitude'],
                                          self.data.loc[j, 'pickup_longitude'], self.data.loc[j, 'pickup_latitude'])
            time_between = timedelta(hours=dist_between / self.mph)
            output = self.data.loc[i, 'dropoff_datetime'] - \
                self.data.loc[i, 'pickup_datetime'] + time_between
        return self.to_minutes(output)

    # Problem with Time Windows

    def problem_window(self, time_window=0):
        # Variables
        x = cp.Variable(self.num_arcs, boolean=True)
        f = cp.Variable(self.n)
        t = cp.Variable(self.n)

        A_in, A_out, e, p = self.flow_vars()
        A = A_in - A_out
        B, d = self.time_vars(time_window)

        C = np.transpose(A[:self.n, ])

        A[self.n+1:, ] = -A[self.n+1:, ]
        # Network Flow Constraints
        constraints = [A @ x <= e, f >= 0, f <= 1]
        inflow = A_in @ x
        outflow = A_out @ x
        constraints += [inflow[:self.n] == f, outflow[:self.n] == f]

        # Time Window Constraints
        t_min, t_max = self.time_cons(time_window)
        constraints += [t_min[:self.n] <= t, t <= t_max[:self.n]]
        constraints += [B @ x + C @ t <= d]
        # constraints += [np.ones(self.n)@(t - t_min[:self.n]) == 0]

        profit = np.transpose(p) @ x
        # - cp.norm(t[:self.n] - t_min[:self.n], 1)
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return x, problem, t, t_min, t_max, f, self.Tmatrix

    def problem_window_10(self, time_window=0):
        # Variables
        x = []
        f = []
        t = []
        for i in range(10):
            x.append(cp.Variable(self.num_arcs, boolean=True))
            f.append(cp.Variable(self.n))
            t.append(cp.Variable(self.n))

        eps = []
        for i in range(10):
            # eps.append(np.zeros(self.num_arcs))
            eps.append(np.random.normal(0, 8, self.num_arcs))

        A_in, A_out, e, p = self.flow_vars()
        A = A_in - A_out
        B, d = self.time_vars(time_window)

        C = np.transpose(A[:self.n, ])

        A[self.n+1:, ] = -A[self.n+1:, ]
        # Network Flow Constraints
        constraints = []
        t_min, t_max = self.time_cons(time_window)

        for i in range(10):
            constraints += [A @ x[i] <= e, f[i] >= 0, f[i] <= 1]
            inflow = A_in @ x[i]
            outflow = A_out @ x[i]
            constraints += [inflow[:self.n] == f[i], outflow[:self.n] == f[i]]
            constraints += [t_min[:self.n] <= t[i], t[i] <= t_max[:self.n]]
            constraints += [(B + cp.diag(eps[i])) @ x[i] + C @ t[i] <= d]

        profit = 0
        for i in range(10):
            profit += np.transpose(p) @ x[i]
        # - cp.norm(t[:self.n] - t_min[:self.n], 1)
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return x, problem, t, t_min, t_max, f, self.Tmatrix, eps

    # Problem with Times as Parameters

    def time_params(self, t, time_window):
        b = []
        d = []
#         d = cp.Variable(self.num_arcs)
        time_cons = []
        t_min = t
        t_max = t
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            b.append(t_max[i] - t_min[j] + self.T(i, j))
            d.append(t_max[i] - t_min[j])
#             time_cons += [d[a] == t_max[i] - t_min[j]]
        return cp.hstack(b), cp.hstack(d), time_cons

    def time_params1(self, time_window, eps):
        b = []
        d = []
#         d = cp.Variable(self.num_arcs)
        time_cons = []
        t_min, t_max = self.time_cons(time_window)
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            b.append(t_max[i] - t_min[j] + self.T(i, j) + eps[a])
            d.append(t_max[i] - t_min[j])
#             time_cons += [d[a] == t_max[i] - t_min[j]]
        return cp.hstack(b), cp.hstack(d), time_cons

    def T_params(self, t, time_window):
        b = np.zeros((self.num_arcs, self.num_arcs))
#         d = cp.Variable(self.num_arcs)
        time_cons = []
        for a in range(self.num_arcs):
            i = self.arcs[a][0]
            j = self.arcs[a][1]
            b[a, a] = self.T(i, j)
        return b

    def problem_param(self, waiting_cost=0, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs)
        f = cp.Variable(self.n)
        t = cp.Parameter(self.num_nodes)

        A_in, A_out, e, p = self.flow_vars()
        A = A_in - A_out
        A[self.n+1:, ] = - A[self.n+1:, ]
        b, d, constraints = self.time_params(t, time_window)
        C = np.transpose(A[:self.n, ])

        constraints += [A @ x <= e, x >= 0, x <= 1, f >= 0, f <= 1]
        inflow = A_in @ x
        outflow = A_out @ x
        constraints += [inflow[:self.n] == f, outflow[:self.n] == f]

        constraints += [cp.multiply(b, x) + C @ t[:self.n] <= d]

        profit = np.transpose(p) @ x
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return t, x, p, problem

    def problem_window2(self, time_window=0):
        # Variables
        x = cp.Variable(self.num_arcs)
        f = cp.Variable(self.n)
        t = cp.Variable(self.n)

        A_in, A_out, e, p = self.flow_vars()
        A = A_in - A_out
        B, d = self.time_vars(time_window)
        C = np.transpose(A[:self.n, ])

        A[self.n+1:, ] = -A[self.n+1:, ]
        # Network Flow Constraints
        constraints = [A @ x <= e, f >= 0, f <= 1, x >= 0, x <= 1]
        inflow = A_in @ x
        outflow = A_out @ x
        constraints += [inflow[:self.n] == f, outflow[:self.n] == f]

        # Time Window Constraints
        t_min, t_max = self.time_cons(time_window)
        constraints += [t_min[:self.n] <= t, t <= t_max[:self.n]]
        constraints += [B @ x + C @ t <= d]
        # constraints += [np.ones(self.n)@(t - t_min[:self.n]) == 0]

        profit = np.transpose(p) @ x - cp.norm(t[:self.n] - t_min[:self.n], 1)
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return x, problem, t, t_min, t_max, f, self.Tmatrix, p

    def problem_param_time(self, waiting_cost=0, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs)
        f = cp.Variable(self.n)
        t = cp.Parameter(self.num_nodes)

        t_min, t_max = self.time_cons(time_window)
        A_in, A_out, e, p = self.flow_vars()
        A = A_in - A_out
        A[self.n+1:, ] = - A[self.n+1:, ]
        b, d, constraints = self.time_params(t, time_window)
        C = np.transpose(A[:self.n, ])

        constraints += [A @ x <= e, x >= 0, x <= 1, f >= 0, f <= 1]
        inflow = A_in @ x
        outflow = A_out @ x
        constraints += [inflow[:self.n] == f, outflow[:self.n] == f]
        constraints += [cp.multiply(b, x) + C @ t[:self.n] <= d]

        profit = np.transpose(
            p) @ x - (t[:self.n] - t_min[:self.n])@f
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return t, x, f, problem

    def problem_param_compact(self, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs + self.n)
        t = cp.Parameter(self.num_nodes)
        #eps = cp.Parameter(self.num_arcs)

        A_begin, A_end, e, p = self.flow_vars()
        A = A_begin - A_end
        b, d, constraints = self.time_params(t, time_window)

        p = np.concatenate((p, np.zeros(self.n)))
        A[self.n+1:, ] = -A[self.n+1:, ]
        Atemp = spa.bmat([[A, spa.dok_matrix((self.num_nodes, self.n))]])
        G = spa.bmat([[Atemp], [spa.identity(self.num_arcs + self.n)]])
        h = np.concatenate(
            (np.zeros(self.n), self.num_taxis*np.ones(2), np.ones(self.num_arcs + self.n)))
        constraints += [G @ x <= h, x >= 0]
        A_final = spa.bmat([[A_begin[:self.n, ], -spa.identity(self.n)],
                            [A_end[:self.n, ], -spa.identity(self.n)]])

        constraints += [A_final@x == 0]
        constraints += [cp.multiply(b, x[:self.num_arcs]) <= 0]
        C = np.transpose(A[:self.n, ])
        B = self.T_params(t, time_window)
        profit = np.transpose(p) @ x
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return t, x, p, problem, constraints, G, h, A_final, b, C, B

    def problem_param_stoch(self, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs + self.n)
        t = cp.Parameter(self.num_nodes)
        rho = cp.Parameter()
        t_min, t_max = self.time_cons(time_window)

        A_begin, A_end, e, p = self.flow_vars()
        A = A_begin - A_end
        b, d, constraints = self.time_params(t, time_window)

        C = np.transpose(A.copy())

        p = np.concatenate((p, np.zeros(self.n)))
        A[self.n+1:, ] = -A[self.n+1:, ]
        Atemp = spa.bmat([[A, spa.dok_matrix((self.num_nodes, self.n))]])
        G = spa.bmat([[Atemp], [spa.identity(self.num_arcs + self.n)]])
        h = np.concatenate(
            (np.zeros(self.n), self.num_taxis*np.ones(2), np.ones(self.num_arcs + self.n)))
        constraints += [G @ x <= h, x >= 0]
        A_final = spa.bmat([[A_begin[:self.n, ], -spa.identity(self.n)],
                            [A_end[:self.n, ], -spa.identity(self.n)]])

        constraints += [A_final@x == 0]
        B = self.T_params(t, time_window)

        # constraints += [cp.SOC(-b[i]*x[i], rho*C[i].T*x[i])
        #               for i in range(self.num_arcs)]

        constraints += [cp.SOC(-b@cp.diag(x[:self.num_arcs]),
                               rho*cp.diag(x[:self.num_arcs])@C, axis=1)]

        # constraints += [cp.multiply(b + rho*np.sqrt(2),
        #                           x[:self.num_arcs]) <= 0]

        profit = np.transpose(
            p) @ x - (t[:self.n] - np.array(t_min[:self.n]))@x[self.num_arcs:]
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return t, x, p, rho, problem, C

    def problem_param_barrier(self, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs + self.n)
        t = cp.Parameter(self.num_nodes)
        w = cp.Parameter(self.num_arcs)
        mu = cp.Parameter(nonneg=True)
        # u1 = cp.Variable(self.num_arcs + 2*self.n + 2)
        # u2 = cp.Variable(self.num_arcs)

        A_begin, A_end, e, p = self.flow_vars()
        A = A_begin - A_end
        b, d, constraints = self.time_params(t, time_window)

        p = np.concatenate((p, np.zeros(self.n)))
        A[self.n+1:, ] = -A[self.n+1:, ]
        Atemp = spa.bmat([[A, spa.dok_matrix((self.num_nodes, self.n))]])
        AA = spa.bmat([[Atemp], [spa.identity(self.num_arcs + self.n)]])
        AB = np.concatenate(
            (np.zeros(self.n), self.num_taxis*np.ones(2), np.ones(self.num_arcs + self.n)))
        # constraints += [AA @ x <= AB, x >= 0]
        BB = spa.bmat([[A_begin[:self.n, ], -spa.identity(self.n)],
                      [A_end[:self.n, ], -spa.identity(self.n)]])
        # constraints += [x >= 0, x <= 1]
        constraints += [BB@x == 0]
        # constraints += [cp.multiply(b, x[:self.num_arcs]) <= 0]

        profit = -w @ x[: self.num_arcs]
        barrier = -cp.sum(cp.log(AB - (AA @ x))) - cp.sum(
            cp.log(-cp.multiply(b, x[:self.num_arcs]))) - cp.sum(cp.log(x))
        objective = cp.Minimize(profit + mu*barrier)
        problem = cp.Problem(objective, constraints)

        return t, x, p, w, problem, constraints, AB, AA, b, mu

    def problem_param_compact_weight(self, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs + self.n)
        t = cp.Parameter(self.num_nodes)
        w = cp.Parameter(self.num_arcs)

        t_min, t_max = self.time_cons(time_window)
        A_begin, A_end, e, p = self.flow_vars()
        A = A_begin - A_end
        b, d, constraints = self.time_params(t, time_window)

        p = np.concatenate((p, np.zeros(self.n)))
        A[self.n+1:, ] = -A[self.n+1:, ]
        Atemp = spa.bmat([[A, spa.dok_matrix((self.num_nodes, self.n))]])
        AA = spa.bmat([[Atemp], [spa.identity(self.num_arcs + self.n)]])
        AB = np.concatenate(
            (np.zeros(self.n), self.num_taxis*np.ones(2), np.ones(self.num_arcs + self.n)))
        constraints += [AA @ x <= AB, x >= 0]
        BB = spa.bmat([[A_begin[:self.n, ], -spa.identity(self.n)],
                       [A_end[:self.n, ], -spa.identity(self.n)]])

        constraints += [BB@x == 0]
        constraints += [cp.multiply(b, x[:self.num_arcs]) <= 0]

        profit = w @ x[:self.num_arcs] - 0.5*cp.norm(t - t_min, 2)
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return t, x, w, problem

    def problem_param_compact_t(self, time_window=0):
        # Variables and Parameter
        x = cp.Variable(self.num_arcs + self.n)
        t = cp.Parameter(self.num_nodes)

        t_min, t_max = self.time_cons(time_window)
        A_begin, A_end, e, p = self.flow_vars()
        A = A_begin - A_end
        b, d, constraints = self.time_params(t, time_window)

        p = np.concatenate((p, np.zeros(self.n)))
        A[self.n+1:, ] = -A[self.n+1:, ]
        Atemp = spa.bmat([[A, spa.dok_matrix((self.num_nodes, self.n))]])
        AA = spa.bmat([[Atemp], [spa.identity(self.num_arcs + self.n)]])
        AB = np.concatenate(
            (np.zeros(self.n), self.num_taxis*np.ones(2), np.ones(self.num_arcs + self.n)))
        constraints += [AA @ x <= AB, x >= 0]
        BB = spa.bmat([[A_begin[:self.n, ], -spa.identity(self.n)],
                       [A_end[:self.n, ], -spa.identity(self.n)]])

        constraints += [BB@x == 0]
        constraints += [cp.multiply(b, x[:self.num_arcs]) <= 0]

        profit = np.transpose(
            p) @ x - cp.norm(t[:self.n] - t_min[:self.n], 2)
        objective = cp.Minimize(-profit)
        problem = cp.Problem(objective, constraints)
        return t, x, p, problem

    def problem_param_dual(self, time_window=0):
        # Variables and Parameter
        y = cp.Variable(self.num_nodes + self.num_arcs + self.n)
        z = cp.Variable(2*self.n)
        v = cp.Variable(self.num_arcs)
        t = cp.Parameter(self.num_nodes)

        A_begin, A_end, e, p = self.flow_vars()
        A = A_begin - A_end
        b, d, constraints = self.time_params(t, time_window)
        p = np.concatenate((p, np.zeros(self.n)))
        A[self.n + 1:, ] = -A[self.n + 1:, ]

        Atemp = spa.bmat([[A, spa.dok_matrix((self.num_nodes, self.n))]])
        AA = spa.bmat([[Atemp], [spa.identity(self.num_arcs + self.n)]])
        AB = np.concatenate(
            (np.zeros(self.n), self.num_taxis*np.ones(2), np.ones(self.num_arcs + self.n)))

        BB = spa.bmat([[A_begin[:self.n, ], -spa.identity(self.n)],
                       [A_end[:self.n, ], -spa.identity(self.n)]])

        constraints += [(np.transpose(AA) @ y)[:self.num_arcs] + (np.transpose(BB)@z)
                        [:self.num_arcs] + cp.multiply(b, v) - p[:self.num_arcs] >= 0]
        constraints += [(np.transpose(AA)@y)[self.num_arcs:] +
                        (np.transpose(BB)@z)[self.num_arcs:] >= 0]

        constraints += [y >= 0, v >= 0]

        profit = np.transpose(AB) @ y
        objective = cp.Minimize(profit)
        problem = cp.Problem(objective, constraints)
        return t, y, z, v, AB, problem

    # Helper Functions

    def to_minutes(self, td):
        return td.total_seconds() / 60

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 3956
        return c * r
