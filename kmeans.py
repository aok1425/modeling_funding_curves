# figure out how to add axes labels to J curve
# stuck on line 80. i know w/o classes how to get methods to run through
# the same data over and over, as it gets constantly updated. but i can't
# seem to get it to make a variable w/in a method be equal to calling
# another method in that class.

import numpy as np
import pandas as pd

def test():
    from test_for_kmeans import grab_values
    values = grab_values.values
    a = KMeans(max_iters=7, tries=15)
    
    a.fit(values=values)
    print '\n\nFitting KMeans for one value of K works!\n\n'

    a.plotj(maxK=4)  # must do after fit() in order to load the values
    print '\n\nFitting KMeans for multiple values of K works!\n\n'

    a.find_centroid_for_each()
    print '\n\nFinding the group for each dot for the lowest-cost centroid position works!\n\n'

class KMeans(object):
    def __init__(self, max_iters, tries):
        """How many times to cycle the new centroids mean and closest points in each K-means iteration?
        How many K-means iterations to try to find the one w/the lowest J, assuming random initial centroids?"""

        if max_iters < 3:
            print 'K must be 3 or more.'
        self.max_iters = max_iters
        self.tries = tries

    # Part 2: What are the initial centroids?
    def evencentroids(self):  # tested for 1D, 2D, 3D
        x = self.x
        K = self.K
        dim = x.shape[1]
        therange = (x.max(axis=0) - x.min(axis=0)) / float(K + 1)
        c = np.arange(1, K + 1)
        step1 = c.reshape(K, 1) * therange
        step2 = step1 + x.min(axis=0)
        step3 = step2.reshape(K, 1, dim)
        self.centroids = step3

    def randomcentroids(self):  # tested for 1D, 2D, 3D
        step1 = np.random.permutation(self.x)
        step2 = step1[:self.K]

        self.centroids = step2.reshape(self.K, 1, self.dim)

    # Part 3: Functions to iterate through the centroids and closest points.
    def closestcentroids(self):
        centroids = self.centroids
        X = self.x
        dim = self.dim

        if dim == 1:
            # works only for 1D data, not n*D like Octave script
            dist = (centroids - X) ** 2
            self.index = dist.argmin(axis=0)
        # X must have dimensions m,n. Centroids must have dimensions m,1,n.
        # It's a 3D vector to keep it in one 'row.'
        elif dim > 1:
            # axis=2 for 3d through trial-and-error. same w/T!
            dist = np.linalg.norm(X - centroids, axis=2).T
            self.index = dist.argmin(axis=1)
        else:
            print 'Can\'t understand how many dimensions this data has!'

    def newcentroids(self):
        centroids = self.centroids
        idx = self.index
        X = self.x
        dim = X.shape[1]

        for i in range(centroids.shape[0]):
            centroids[i][0] = X[idx == i].mean(axis=0)

        self.centroids = centroids
    # sometimes X[idx==i] is empty when centroids are too close to each other.
    # i just skip this calculation in run_k_means, using pass.

    # Part 4: Running the K-means clustering.
    def run_k_means(self):
        """return centroid position, how many for each, and cost"""
        centroids = self.centroids

        for i in range(self.max_iters):
            self.closestcentroids()
            self.newcentroids()

        J = 0
        X = self.x
        m = len(X)
        idx = self.index
        K = self.K
        dim = X.shape[1]

        for num in range(K):
            # find the index of all entries where idx==n
            indexentries = np.nonzero(idx == num)[0]
            # the values in X that have the index in indesxentries
            values = X[indexentries]
            # using one of the K centroids to do the calculation. K<=2 doesn't
            # work here for some reason.
            centroid = centroids[num, 0]
            J += np.sum((values - centroid) ** 2)

        return [centroids.reshape((1, K, dim)), [X[idx == k].size for k in range(K)], J / m]

    # Part 5: Computing the cost.
    def compute_cost(self, index, X):
        J = 0
        m = len(X)
        X = self.x

        for num in range(K):
            # find the index of all entries where idx==n
            indexentries = np.nonzero(index == num)[0]
            # the values in X that have the index in indesxentries
            values = X[indexentries]
            # using one of the K centroids to do the calculation
            centroid = centroids[0, num]
            J += np.sum((values - centroid) ** 2)

        return J / m

    # Part 6: Computing the centroids multiple times to find the one w/lowest cost.
    def k_means_multiple(self, K):
        """How many centroids K to cluster the data with? The algorithm will cycle through self.max_iters times for each try, self.tries."""
        self.K = K
        table = []

        for numberoftimes in range(self.tries):
            self.randomcentroids()
            try:
                atry = self.run_k_means()
                table.append(atry)
            except ValueError:
                pass

        c = ['centroid position', 'how many for each', 'J']

        self.table = pd.DataFrame(table, columns=c).sort_index(by=['J']).head()

    # Part 7: Trying multiple # of centroids, or K, and plotting the cost J for each.
    def plotj(self, maxK):
        """Plots the cost for each number of centroids K to find the optimal number of centroids.
        How many different Ks to calculate, in order to find the right number of centroids?"""

        list = []

        for k in range(2, maxK + 1):
            print 'Calculating for K={}'.format(k)
            self.k_means_multiple(k)

            table = self.table
            table = table.reset_index()
            table = table.drop('index', 1)
            iwant = table.ix[0]

            J = iwant['J']
            list.append([k, J])

        toplot = pd.DataFrame(list)
        toplot = toplot.set_index(0)

        ax = toplot.plot(
            legend=False, ylim=0, xlim=2, xticks=range(2, maxK + 1))
        ax.set_xlabel('Number of centroids')
        ax.set_ylabel('Cost of centroids and their points')

    def fit(self, values):
        self.x = values

        try:
            self.dim = self.x.shape[1]
        except:
            self.dim = 1

        self.K = int(raw_input('How many centroids? '))
        self.randomcentroids()

        print 'Running K-means {} times, moving the {} centroids a max of {} on each try...'.format(self.tries, self.K, self.max_iters)
        self.k_means_multiple(self.K)
        print '\n'
        print self.table

    def find_centroid_for_each(self):
    	"""Which entries belong to which groups?"""

        step1 = self.table['centroid position'].reset_index().ix[0]
        step2 = np.array(step1)[1]
        self.centroids = step2.reshape(self.K, 1, self.dim)
        self.closestcentroids()

        return self.index