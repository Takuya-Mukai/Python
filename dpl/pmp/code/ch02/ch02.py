import numpyas as np


class Perseptron:
    """
    perseptron classifier

    parameter
    ----------------
    eta : float
      learning rate
    n_iter : int
      training number of trainig data
    ramdom_state : int
      random number seed for initialize the weight

    attribution
    ----------------
    w_ : one dimensional array
      weight after applied
    b_ : scalar
      bias unit after applied
    errors_ : list
      Number of misclassifications in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X, y):
        """
        fit to train data

        parameter
        ---------------
        X: {data structure like array}, shape = [n_examples, n_features]
          train vector: n_example is number of train_data, n_feature is number of features
        y: data structure like array, shape = [n_examples]
          objective variable

        return value
        ---------------
        self: object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):            # Iterate over training data
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                self.errors += int(update != 0.0)   # add error
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        # calculate total input
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        # return class rabel after one step
        return np.where(self.net_input(X) >= 0.0, 1,0)    # if net_input >= 0, return 1, else return 0




