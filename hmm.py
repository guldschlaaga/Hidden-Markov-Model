# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.


import argparse
import numpy as np

from nlputil import *  # utility methods for working with text


# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size
#
# Note: You may want to add fields for expectations.
class HMM:
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')

    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.
    def __init__(self, num_states, vocab_size):
        self.pi = (np.zeros(num_states, dtype=np.longdouble) + 1) / num_states
        self.transitions = (np.zeros((num_states, num_states), dtype=np.longdouble) + 1) / num_states
        self.emissions = (np.zeros((num_states, vocab_size), dtype=np.longdouble) + 1) / 4
        self.num_states = num_states
        self.vocab_size = vocab_size
        pass

    def __init__(self, num_states, vocab_size, dataset):
        self.pi = (np.zeros(num_states, dtype=np.longdouble) + 1) / num_states
        self.transitions = (np.zeros((num_states, num_states), dtype=np.longdouble) + 1) / num_states
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.emissions = np.zeros((num_states, vocab_size), dtype=np.longdouble)
        self.emission_helper(dataset)

        pass

    def emission_helper(self, dataset):
        count = np.zeros(self.vocab_size, dtype=int)
        sum = 0
        for sample in dataset:
            for token in sample:
                token = int(token)
                count[token - 1] = count[token - 1] + 1
                sum = sum + 1
        for k in range(self.num_states):
            for i in range(self.vocab_size):
                self.emissions[k][i] = count[i] / (sum)

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset, samples):
        total = 0
        used = 0
        for n in range(samples):
            # if n % 500 == 0:
            #     print(n)
            if len(dataset[n]) < 1000:
                used = used + 1
                total = total + self.loglikelihood_helper(dataset[n])
        print(used)
        return total / used

    # return the loglikelihood for a single sequence (numpy matrix)
    # noinspection PyPep8Naming
    def loglikelihood_helper(self, sample):
        total = np.longdouble(0.0)
        if np.size(sample) == 0.0:
            return 0
        atable = self.alpha(sample)
        for k in range(self.num_states):
            total = total + atable[np.size(sample) - 1][k]
        total = np.log(total)
        return total

    def alpha(self, sample):
        table = np.zeros((np.size(sample), self.num_states), dtype=np.longdouble)

        K = self.num_states

        for k1 in range(K):
            table[0][k1] = self.pi[k1] * self.emissions[k1, int(sample[0]) - 1]

        for n in range(1, np.size(sample)):  # O(NK^2)
            for k2 in range(K):

                scale = self.emissions[k2][int(sample[n] - 1)]
                total = np.longdouble(0.0)
                for k1 in range(K):
                    total = total + table[n - 1][k1] * self.transitions[k1][k2]

                table[n][k2] = scale * total

        return table

    def alphaplus(self, newsample, table):

        K = self.num_states

        n = np.size(table, axis=0)
        newtable = np.zeros((1, self.num_states), dtype=np.longdouble)

        for k2 in range(K):

            scale = self.emissions[k2][int(newsample[n] - 1)]
            total = np.longdouble(0.0)
            for k1 in range(K):
                total = total + table[n - 1][k1] * self.transitions[k1][k2]

            newtable[0][k2] = scale * total

        return np.append(table, newtable, axis=0)

    def beta(self, sample):
        table = np.zeros((np.size(sample), self.num_states), dtype=np.longdouble)

        K = self.num_states

        for k in range(K):
            table[np.size(sample) - 1][k] = 1.0

        for n in reversed(range(np.size(sample) - 1)):  # O(NK^2)

            for k1 in range(K):
                total = np.longdouble(0.0)

                for k2 in range(K):
                    total = total + table[n + 1][k2] * self.transitions[k1][k2] * self.emissions[
                        k2, int(sample[n + 1] - 1)]

                table[n][k1] = total

        return table

    def eata(self, sample, atable, btable):
        # takes in a full alpha table for the dataset and a full beta table, returns NxKxK table of eata
        # values for Zn-1 -> Zn aka (Zn | Zn-1)

        etable = np.zeros((len(sample), self.num_states, self.num_states), dtype=np.longdouble)
        pX = np.longdouble(0.0)

        for k in range(self.num_states):
            pX = pX + atable[np.size(sample) - 1][k]
        for n in range(1, len(sample)):
            for k1 in range(self.num_states):

                for k2 in range(self.num_states):
                    etable[n][k1][k2] = \
                        atable[n - 1][k1] * self.emissions[k2][int(sample[n]) - 1] * self.transitions[k1][k2] * \
                        btable[n][k2] / pX

        return etable

    def gamma(self, sample, atable, btable):
        # takes in a full alpha table for the dataset and a full beta table
        # returns NxK table of gamma values for p(Zn | X(n))

        gtable = np.zeros((len(sample), self.num_states), dtype=np.longdouble)
        pX = np.longdouble(0.0)

        for k in range(self.num_states):
            pX = pX + atable[np.size(sample) - 1][k]

        for n in range(np.size(sample)):
            for k in range(self.num_states):
                gtable[n][k] = atable[n][k] * btable[n][k] / pX

        return gtable

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset):
        # E step

        gammas = []
        eatas = []
        for n in range(len(dataset)):
            if n % 500 == 0:
                print(n)
            atable = self.alpha(dataset[n])
            btable = self.beta(dataset[n])
            eatas.append(self.eata(dataset[n], atable, btable))
            gammas.append(self.gamma(dataset[n], atable, btable))

        # M step

        for k in range(self.num_states):
            total = np.longdouble(0.0)

            for r in range(len(dataset)):
                total = total + gammas[r][0][k]

            self.pi[k] = total / len(dataset)

        for k1 in range(self.num_states):
            for k2 in range(self.num_states):
                totalr1 = np.longdouble(0.0)
                totalr2 = np.longdouble(0.0)
                for r in range(len(dataset)):
                    totaln1 = np.longdouble(0.0)
                    totaln2 = np.longdouble(0.0)
                    for n in range(len(dataset[r]) - 1):
                        totaln1 = totaln1 + eatas[r][n][k1][k2]
                        totaln2 = totaln2 + gammas[r][n][k1]
                    totalr1 = totalr1 + totaln1
                    totalr2 = totalr2 + totaln2
                self.transitions[k1][k2] = totalr1 / totalr2
        for k in range(self.num_states):

            sums = np.zeros(self.vocab_size, dtype=np.longdouble)

            total = 0.0

            for r in range(len(dataset)):

                for n in range(len(dataset[r])):
                    sums[int(dataset[r][n]) - 1] += gammas[r][n][k]

                    total = total + gammas[r][n][k]
            for v in range(self.vocab_size):
                self.emissions[k][v] = sums[v] / total

        pass

    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        xtable = np.zeros(self.vocab_size, dtype=np.longdouble)
        atable = self.alpha(sample)

        for i in range(steps):
            pX = np.longdouble(0.0)

            for k in range(self.num_states):
                pX = pX + atable[np.size(sample) - 1][k]
            max = 0.0

            selection = 0
            for v in range(self.vocab_size):
                big = np.longdouble(0.0)
                for k2 in range(self.num_states):
                    total = np.longdouble(0.0)
                    for k1 in range(self.num_states):
                        total = total + self.transitions[k1][k2] * atable[np.size(sample) - 1][k1]
                    big = big + self.emissions[k2][v] * total
                xtable[v] = big / pX
                if v < 100:

                    print(xtable[v])
                if xtable[v] > max:

                    selection = v + 1
                    max = xtable[v]

            sample = np.append(sample, selection)

            atable = self.alphaplus(sample, atable)


        return sample


def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--train_path', default=None, help='Path to the training data directory.')
    parser.add_argument('--dev_path', default=None, help='Path to the development data directory.')
    parser.add_argument('--epsilon', default=8, help='convergence signifier.')
    parser.add_argument('--subsample', default=5000, help='max number of sequences to include in dataset.')
    parser.add_argument('--max_iters', type=int, default=10, help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--hidden_states', type=int, default=6,
                        help='The number of hidden states to use. (default 10)')
    args = parser.parse_args()

    paths = [args.train_path]
    print("Begin loading vocab... ", end='')
    sys.stdout.flush()
    begin = time()
    vocab = build_vocab_words(paths)
    end = time()
    print('done in', end - begin, 'seconds.  Found', len(vocab), 'unique tokens.')
    print('Begin loading all data and converting to ints... ', end='')
    sys.stdout.flush()
    begin = time()
    data = load_and_convert_data_words_to_ints(paths, vocab)
    end = time()
    print('done in', end - begin, 'seconds.')

    biggest = 0
    average = 0.0
    for sample in data:
        if len(sample) > biggest:
            biggest = len(sample)
        average = average + len(sample)
    average = average / len(data)
    print("biggest is: " + str(biggest))
    print("average is: " + str(average))

    print('Begin initializing model... ', end='')
    begin = time()
    model = HMM(int(args.hidden_states), len(vocab), data)
    end = time()
    print('done in', end - begin, 'seconds.')
    prev = model.loglikelihood(data, int(args.subsample))
    print(prev)

    used = 0
    udata = []
    for n in range(int(args.subsample)):

        if len(data[n]) < 1000 and len(data[n] > 0) and len(data) > n:
            udata.append(data[n])
            used = used + 1

    print('Begin doing em step... ')

    begin = time()
    epsilon = np.longdouble(np.power(10.0, -1.0 * float(args.epsilon)))
    for i in range(int(args.max_iters)):
        model.em_step(udata)
        next = model.loglikelihood(data, int(args.subsample))
        print(next)
        if np.abs(prev - next) < epsilon:
            print("Converged early")
            break
        prev = next
    end = time()

    completed = model.complete_sequence(udata[2][0:49], 5)
    print(str(udata[2][49]) + " vs " + str(completed[49]) + ", " + str(udata[2][50]) + " vs " + str(completed[50]))
    print(completed)

    print('Done in', end - begin, 'seconds.')

    print("done")

    # OVERALL PROJECT ALGORITHM:
    # 1. load training and testing data into memory
    # using training data ONLY
    # 2. build vocabulary
    #
    # 3. instantiate an HMM with given number of states -- initial parameters can
    #    be random or uniform for transitions and inital state distributions,
    #    initial emission parameters could bea uniform OR based on vocabulary
    #    frequency (you'll have to count the words/characters as they occur in
    #    the training data.)
    #
    # 4. output initial loglikelihood on training data and on testing data
    #
    # 5+. use EM to train the HMM on the training data,
    #     output loglikelihood on train and test after each iteration
    #     if it converges early, stop the loop and print a message


if __name__ == '__main__':
    main()
