import json
import numpy as np


class MyHmm(object):  # base class for different HMM models
    def __init__(self, model_name):
        # model is (A, B, pi) where A = Transition probs, B = Emission Probs, pi = initial distribution
        self.model = json.loads(open(model_name).read())
        self.A = self.model["A"]
        self.states = self.A.keys()  # get the list of states
        self.N = len(self.states)  # number of states of the model
        self.B = self.model["B"]
        s = []
        for k, v in self.B.items():
            s.extend(v.keys())
        s = set(s)
        self.symbols = s  # get the list of symbols, assume that all symbols are listed in the B matrix
        self.M = len(self.symbols)  # number of states of the model
        self.pi = self.model["pi"]
        return

    def backward(self, obs):
        self.bwk = [{} for t in range(len(obs))]
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states: self.bwk[T - 1][y] = 1
        for t in reversed(range(T - 1)):
            for y in self.states:
                self.bwk[t][y] = sum(
                    (self.bwk[t + 1][y1] * self.A[y][y1] * self.B[y1][obs[t + 1]]) for y1 in self.states)
        prob = sum((self.pi[y] * self.B[y][obs[0]] * self.bwk[0][y]) for y in self.states)
        return prob

    def forward(self, obs):
        self.fwd = [{}]
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob

    def maximum_liklihood(self, data):
        # compute pi
        for y in self.pi:
            counter = 0
            all = 0
            for d in data:
                all += 1
                if d[0] == y:
                    counter += 1
            self.pi[y] = counter / all
        # compute A
        for y in self.A:
            today = y
            for tomorrow in self.A[today]:
                counter = 0
                all = 0
                for i in range(len(data) - 1):
                    if data[i][0] == today :
                        all+=1
                        if data[i + 1][0] == tomorrow:
                            counter += 1
                self.A[today][tomorrow] = counter / all
        # compute B
        for y in self.B:
            for umbrella in self.B[y]:
                counter = 0
                all = 0
                for i in range(len(data)):
                    if data[i][0] == y:
                        all += 1
                        if data[i][1] == umbrella:
                            counter += 1
                self.B[y][umbrella] = counter / all

    def forward_backward(self, obs, iteration):  # returns model given the initial model and observations
        for d in range(iteration):
            landa = [{} for t in range(len(obs))]
            phi = [{} for t in range(len(obs) - 1)]
            # Expectation step : compute landa and phi
            p_obs = self.forward(obs)
            self.backward(obs)
            for t in range(len(obs)):
                for y in self.states:
                    landa[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                    if t == 0: self.pi[y] = landa[t][y]
                    if t == len(obs) - 1: continue
                    phi[t][y] = {}
                    for y1 in self.states:
                        phi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][
                            y1] / p_obs
            # Maximization step : compute A and B
            for y in self.states:
                # compute A
                for y1 in self.states:
                    val = sum([phi[t][y][y1] for t in range(len(obs) - 1)])  #
                    val /= sum([landa[t][y] for t in range(len(obs) - 1)])
                    self.A[y][y1] = val
                # compute B
                for k in self.symbols:
                    val = 0.0
                    for t in range(len(obs)):
                        if obs[t] == k: val += landa[t][y]
                    val /= sum([landa[t][y] for t in range(len(obs))])
                    self.B[y][k] = val
        return

    def print_model(self, iter):
        if iter > 0:
            print("The new model parameters after " + str(iter) + " iteration are: ")
        else:
            print("The new model parameters : ")
        print("A = ", self.A)
        print("B = ", self.B)
        print("P = ", self.pi)
        with open('data' + str(iter) + '.txt', 'w') as outfile:
            self.model["A"] = self.A
            self.model["B"] = self.B
            self.model["pi"] = self.pi
            json.dump(self.model, outfile)
        print()

    def viterbi(self, obs):
        vit = [{}]
        path = {}
        n = 0
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}
            for y in self.states:
                (prob, state) = max((vit[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
            n += 1
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return prob, path[state]


def get_observation(file_obs):
    file = open(file_obs)
    obs = []
    for line in file:
        words = line.split(",")
        val = words[1].split("\n")[0]
        obs.append(val)
    return obs


def get_all_data(file_obs):
    file = open(file_obs)
    obs = []
    for line in file:
        temp = []
        words = line.split(",")
        temp.append(words[0])
        temp.append(words[1].split("\n")[0])
        obs.append(temp)
    return obs


# part a
data = get_all_data("Test2.txt")
print("part a")
print("Learning the model with maximum liklihood for the observations")  # , observations)
hmm1 = MyHmm("equal.json")
hmm1.maximum_liklihood(data)
hmm1.print_model(0)
# part b
observations = get_observation("Test2.txt")
print("part b")
print("Learning the model through Forward-Backward Algorithm for the observations")  # , observations)
iteration = [50]
for iter in iteration:
    hmm2 = MyHmm("equal.json")
    hmm2.forward_backward(observations, iter)
    hmm2.print_model(iter)
# part c
print("part c")
test_obs = ['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
print("observation :", test_obs)
prob, hidden_states = hmm1.viterbi(test_obs)
print("use generated HMM of part a")
print("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)
print("use generated HMM of part b")
prob, hidden_states = hmm2.viterbi(test_obs)
print("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)
