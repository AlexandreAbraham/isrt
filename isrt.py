from recordclass import dataobject
import numpy as np


# export Model, miSVM, maxinst


class Options(dataobject):

    n_trees: int
    n_subfeat: int
    n_thresholds: int
    max_depth: int
    min_samples_leaf: int
    min_samples_split: int
    bagging: float
    lambdas: np.array
    sparse: bool
    epochs: int


class Split(dataobject):

    cost: float
    feature: int
    threshold: float
    selector: np.array

    n_left: int
    n_right: int

    def is_left(self, bag):
        instance, _ = maxinst(bag, self.selector)
        return bag[instance, self.feature] < self.threshold

    def print(self, pre=''):
        print(pre, self.feature, '<', self.threshold)


def maxinst(bag: np.array, w: np.array):
    assert(len(bag.shape) == 2)
    assert(len(w.shape) == 1)
    assert(bag.shape[1] == w.shape[0])

    values = (bag * w).sum(axis=1)
    i_max = values.argmax()
    return i_max, values[i_max]

# function (split::Split)(bag::Matrix{Float32})
# 
#     instance, _ = maxinst(bag, split.selector)
#     return bag[instance, split.feature] < split.threshold
# 
# end

class Node:

    def __init__(self, depth: int, samples: np.array):
        self.depth = depth
        self.samples = samples
        self.is_leaf = True
        self.probability = 0
        self.left = None
        self.right = None
        self.split = None  # Split
        self.parent = None


    def print(self, pre=''):
        if self.is_leaf:
            print(pre, self.probability)
        else:
            self.split.print(pre=pre)
            self.left.print(pre=pre + '  |')
            self.right.print(pre=pre + '  |')


def entropy_loss(V, y, n_pos_samples, n_neg_samples, n_samples, feature, threshold, selector):

    n_left_pos = 0
    n_left_neg = 0

    for ii in range(n_samples):
        if V[ii] < threshold:
            if y[ii]:
                n_left_pos += 1
            else:
                n_left_neg += 1

    n_left = n_left_pos + n_left_neg
    w_left = n_left / n_samples
    p_left_pos = n_left_pos / n_left
    p_left_neg = 1.0 - p_left_pos
    entropy_left = 0. if p_left_neg == 0. else -p_left_neg * np.log2(p_left_neg)
    entropy_left += 0. if p_left_pos == 0. else -p_left_pos * np.log2(p_left_pos)

    n_right_pos = n_pos_samples - n_left_pos
    n_right_neg = n_neg_samples - n_left_neg

    n_right = n_right_pos + n_right_neg
    w_right = n_right / n_samples
    p_right_pos = n_right_pos / n_right
    p_right_neg = 1.0 - p_right_pos
    entropy_right = 0. if p_right_neg == 0.0 else -p_right_neg * np.log2(p_right_neg)
    entropy_right += 0. if p_right_pos == 0.0 else -p_right_pos * np.log2(p_right_pos)

    cost = (w_left * entropy_left) + (w_right * entropy_right)

    return Split(
        cost=cost,
        feature=feature,
        threshold=threshold,
        selector=selector, 
        n_left=n_left,
        n_right=n_right)


def miSVM_update_wb(w, b, t, λ, bag, label, ind):
    
    maxi, maxv = maxinst(bag, w)
    
    η = 1. / (λ * t)
    α = 1. - η * λ

    if (label * (maxv + b)) < 1.:
        β = label * η
        for ii in ind:
            w[ii] = α * w[ii] + β * bag[maxi, ii]
        return α * b + β
    else:
        for ii in ind:
            w[ii] = α * w[ii]
        return α * b


def miSVM(rng,
          X: list,
          Y: np.array,
          λ: float=1.,
          epochs: int=100,
          sparse: bool=True):
   
    labels = (Y.astype(int) * 2) - 1

    n_features = X[0].shape[1]
    b = rng.random()

    if not sparse:
        ind = np.arange(n_features)
    else:
        ind = rng.choice(n_features, size=round(n_features ** .5), replace=False)
    
    w = np.zeros(n_features)
    w[ind] = rng.random(size=ind.shape[0])

    t = 1

    for _ in range(epochs):
        randbags = rng.choice(len(X), size=len(X), replace=True)
        for bb in randbags:
            b = miSVM_update_wb(w, b, t, λ, X[bb], labels[bb], ind)
            t += 1
    
    # Note: b below is never used... Is that a bug top return it?
    return w  # , b


def split(rng, node, X, Y, features, opt, verbose=0):

    Y_node = Y[node.samples]
    n_samples = node.samples.shape[0]
    n_pos_samples = sum(Y_node)
    n_neg_samples = n_samples - n_pos_samples
    node.probability = n_pos_samples / n_samples

    if (node.depth == opt.max_depth
            or node.probability == 1.
            or node.probability == 0.
            or n_samples < opt.min_samples_split
            or n_samples == opt.min_samples_leaf):
        node.is_leaf = True

        if verbose > 0:
            print('No split due to condition 1')

        return

    best_split = None

    X_node = [X[i] for i in node.samples]  # to check
    I = np.empty(n_samples, dtype=int)
    V = np.empty(n_samples)
    selectors = []

    for (ll, λ) in enumerate(opt.lambdas):
        selectors.append(miSVM(rng, X_node, Y_node, λ, opt.epochs, opt.sparse))

    if node.parent is not None:
        selectors.append(node.parent.split.selector)

    for selector in selectors:

        for (ii, bag) in enumerate(X_node):
            I[ii], _ = maxinst(bag, selector)
        
        mtry = 1
        shuffled_features = features.copy()
        rng.shuffle(shuffled_features)

        for feature in shuffled_features:

            minv, maxv = None, None
            for (ii, bag) in enumerate(X_node):
                val = bag[I[ii], feature]
                if minv is None or val < minv:
                    minv = val
                if maxv is None or val > maxv:
                    maxv = val
                V[ii] = val

            if minv == maxv:
                continue

            scale = maxv - minv

            for ii in range(opt.n_thresholds):
                threshold = rng.random() * scale + minv
                split = entropy_loss(V, Y_node, n_pos_samples, n_neg_samples, n_samples, feature, threshold, selector)
                if best_split is None or split.cost < best_split.cost:
                    best_split = split

            if mtry == opt.n_subfeat:
                break
            else:
                mtry += 1

    if (best_split is None
            or best_split.feature == 0
            or best_split.n_left < opt.min_samples_leaf
            or best_split.n_right < opt.min_samples_leaf):
        node.is_leaf = True
        if verbose > 0:
            print('No split due to condition 2.')
            if best_split is not None:
                print('[{}, {}]'.format(best_split.n_left, best_split.n_right))
        return

    node.is_leaf = False
    node.split = best_split

    ll, rr = 0, 0
    left_samples = np.empty(best_split.n_left, dtype=int)
    right_samples = np.empty(best_split.n_right, dtype=int)
    for (ss, bag) in zip(node.samples, X_node):
        if best_split.is_left(bag):
            left_samples[ll] = ss
            ll += 1
        else:
            right_samples[rr] = ss
            rr += 1

    node.left = Node(node.depth + 1, left_samples)
    node.left.parent = node

    node.right = Node(node.depth + 1, right_samples)
    node.right.parent = node

    return


def tree_builder(
        X: list,
        Y: np.array,  # of bool
        opt: Options,
        seed: int=1234):
    rng = np.random.default_rng(seed)
    n_samples, n_features = len(X), X[0].shape[1]
    features = np.arange(n_features)

    samples = np.arange(n_samples)
    if opt.bagging > 0.:
        samples = rng.choice(samples, round(opt.bagging * n_samples))

    root = Node(1, samples)
    stack = [root]
    while len(stack) > 0:
        node = stack.pop()
        split(rng, node, X, Y, features, opt)
        if not node.is_leaf:
            stack.extend([node.left, node.right])

    return root


def score_bag(trees, bag):
    probability = 0.0
    for tree in trees:
        node = tree
        while not node.is_leaf:
            if node.split.is_left(bag):
                node = node.left
            else:
                node = node.right
            node = node
        probability += node.probability

    return probability / len(trees)


class Model:

    def __init__(self,

                 n_trees: int=100,
                 n_subfeat: int=0,
                 n_thresholds: int=1,
                 max_depth: int=-1,
                 min_samples_leaf: int=1,
                 min_samples_split: int=2,
                 bagging: float=0.,
                 lambdas: np.array=np.asarray([1.]),
                 sparse: bool=True,
                 epochs: int=10,
                 seed: int=1234):
        
        assert(n_trees >= 1)
        assert(n_thresholds >= 1)
        assert(min_samples_leaf >= 1)
        assert(min_samples_split >= 1)
        assert(lambdas.shape[0] >= 1)
        assert(bagging >= 0.0)
        assert(epochs >= 1)

        opt = Options(n_trees, n_subfeat, n_thresholds, max_depth, min_samples_leaf, min_samples_split, bagging, lambdas, sparse, epochs)
        self.options = opt

        self.seed = seed


    def fit(self, X: list, Y: np.array):
        assert(len(X) == Y.shape[0])

        if not (1 <= self.options.n_subfeat <= X[0].shape[1]):
            # XXX: this is not a good way to have auto selection of parameter
            self.options.n_subfeat = round(X[0].shape[1] ** .5)

        trees = [None] * self.options.n_trees
        # seeds = abs.(rand(MersenneTwister(seed), Int, n_trees))
        # Threads.@threads for tt in 1:n_trees
        for tt in range(self.options.n_trees):
            trees[tt] = tree_builder(X, Y, self.options, self.seed + tt)

        self.trees_ = trees

    def predict(self, X: list):
        scores = [None] * len(X)

        # Threads.@threads for bb in 1:length(X)
        for i in range(len(X)):
            scores[i] = score_bag(self.trees_, X[i])

        return scores

    def print(self):
        for i, tree in enumerate(self.trees_):
            print('Tree #{}'.format(i))
            tree.print(pre='  |')


# # predictions with instance level distributions
# 
# function (node::Node)(sample::Matrix{Float32}, instdist::Bool)
# 
#     distribution = zeros(Float64, size(sample, 1))
# 
#     while !node.is_leaf
#         instance, _ = maxinst(sample, node.split.selector)
#         distribution[instance] += 1.0
#         node = (sample[instance, node.split.feature] < node.split.threshold) ? node.left : node.right
#     end
# 
#     return node.probability, distribution ./ sum(distribution)
# 
# end
# 
# 
# function (model::Model)(sample::Matrix{Float32}, instdist::Bool)
# 
#     probability = 0.0
#     distribution = zeros(Float64, size(sample, 1))
# 
#     for tree in model.trees
#         tree_probability, tree_distribution = tree(sample, true)
#         distribution .+= tree_distribution
#         probability += tree_probability
#     end
# 
#     return probability / length(model.trees), distribution ./ length(model.trees)
# 
# end
# 
# 
# function (model::Model)(X::Vector{Matrix{Float32}}, instdist::Bool)
# 
#     scores = Vector{Float64}(undef, length(X))
#     distributions = Vector{Vector{Float64}}(undef, length(X))
# 
#     Threads.@threads for bb in 1:length(X)
#         score, distribution = model(X[bb], true)
#         scores[bb] = score
#         distributions[bb] = distribution
#     end
# 
#     return scores, distributions
# 
# end
# 
# end #module
# 