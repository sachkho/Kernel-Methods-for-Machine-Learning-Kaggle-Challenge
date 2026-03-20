
#Kernel SVM + KRR implemented from scratch.

import numpy as np

try:
    import cvxopt, cvxopt.solvers
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    print("WARNING: cvxopt not found. Install with: pip install cvxopt")

def rbf_kernel(X1, X2, gamma=0.01):
    sq1 = np.sum(X1**2, axis=1, keepdims=True)
    sq2 = np.sum(X2**2, axis=1, keepdims=True).T
    return np.exp(-gamma * np.maximum(sq1 + sq2 - 2*X1@X2.T, 0)).astype(np.float64)

def polynomial_kernel(X1, X2, degree=3, coef0=1.0, gamma=1.0):
    return (gamma * X1@X2.T + coef0)**degree

def linear_kernel(X1, X2):
    return (X1 @ X2.T).astype(np.float64)

def histogram_intersection_kernel(X1, X2):
    n, m = X1.shape[0], X2.shape[0]
    K = np.zeros((n, m), dtype=np.float64)
    for i in range(0, n, 256):
        K[i:i+256] = np.minimum(X1[i:i+256][:,None,:], X2[None,:,:]).sum(axis=2)
    return K

def chi2_kernel(X1, X2, gamma=1.0):
    eps = 1e-10
    n, m = X1.shape[0], X2.shape[0]
    K = np.zeros((n, m), dtype=np.float64)
    for i in range(0, n, 128):
        x = X1[i:i+128][:,None,:]
        y = X2[None,:,:]
        K[i:i+128] = np.exp(-gamma * ((x-y)**2 / (x+y+eps)).sum(axis=2))
    return K


class BinaryKernelSVM:
    def __init__(self, C=1.0, kernel_fn=None, tol=1e-5):
        self.C = C
        self.kernel_fn = kernel_fn if kernel_fn else rbf_kernel
        self.tol = tol
        self.alphas = self.support_vectors = self.support_labels = None
        self.bias = 0.0

    def fit(self, X, y):
        assert CVXOPT_AVAILABLE
        n = X.shape[0]
        K  = self.kernel_fn(X, X)
        yy = y.astype(np.float64)
        P  = cvxopt.matrix(np.outer(yy, yy) * K)
        q  = cvxopt.matrix(-np.ones(n))
        G  = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h  = cvxopt.matrix(np.hstack([np.zeros(n), self.C*np.ones(n)]))
        A  = cvxopt.matrix(yy.reshape(1,-1))
        b  = cvxopt.matrix(0.0)
        cvxopt.solvers.options.update({'show_progress':False,'abstol':1e-7,'reltol':1e-6})
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).ravel()
        sv = alphas > self.tol
        self.alphas = alphas[sv]
        self.support_vectors = X[sv]
        self.support_labels  = yy[sv]
        bounded = sv & (alphas < self.C - self.tol)
        ref = bounded if bounded.sum() > 0 else sv
        K_sv = self.kernel_fn(X[ref], self.support_vectors)
        pred = K_sv @ (self.alphas * self.support_labels)
        self.bias = np.mean((yy if bounded.sum()==0 else y)[ref] - pred)
        print(f"    #SVs: {sv.sum()}/{n}")

    def decision_function(self, X):
        return self.kernel_fn(X, self.support_vectors) @ (self.alphas*self.support_labels) + self.bias

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)


#Multiclass SVM using one-vs-rest strategy with BinaryKernelSVM as base classifier.

class MulticlassKernelSVM:
    def __init__(self, C=1.0, kernel_fn=None):
        self.C = C
        self.kernel_fn = kernel_fn if kernel_fn else rbf_kernel
        self.classifiers = {}; self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        print(f"Training {len(self.classes)} binary SVMs (one-vs-rest)...")
        for k, cls in enumerate(self.classes):
            print(f"  Class {cls} ({k+1}/{len(self.classes)})")
            clf = BinaryKernelSVM(C=self.C, kernel_fn=self.kernel_fn)
            clf.fit(X, np.where(y==cls, 1, -1))
            self.classifiers[cls] = clf

    def decision_function(self, X):
        return np.column_stack([self.classifiers[c].decision_function(X) for c in self.classes])

    def predict(self, X):
        return self.classes[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


#KRR

class KernelRidgeClassifier:
    def __init__(self, lam=1.0, kernel_fn=None, nystrom_m=None, seed=42):
        self.lam = lam
        self.kernel_fn = kernel_fn if kernel_fn else rbf_kernel
        self.nystrom_m = nystrom_m; self.seed = seed
        self.alpha = self.X_train = self.X_anchors = self.classes = None

    def fit(self, X, y):
        self.X_train = X; self.classes = np.unique(y)
        n, K_cls = X.shape[0], len(self.classes)
        Y = np.zeros((n, K_cls), dtype=np.float64)
        for i, cls in enumerate(self.classes):
            Y[y==cls, i] = 1.0

        if self.nystrom_m is None or self.nystrom_m >= n:
            print(f"  KRR full: {n}x{n} kernel...")
            K = self.kernel_fn(X, X)
            self.alpha = np.linalg.solve(K + self.lam*np.eye(n), Y)
        else:
            m = self.nystrom_m
            rng = np.random.RandomState(self.seed)
            self.X_anchors = X[rng.choice(n, m, replace=False)]
            print(f"  KRR Nystrom m={m}: {n}x{m} kernel...")
            K_nm = self.kernel_fn(X, self.X_anchors).astype(np.float64)
            K_mm = self.kernel_fn(self.X_anchors, self.X_anchors).astype(np.float64) + 1e-8*np.eye(m)
            A = self.lam*K_mm + K_nm.T@K_nm + 1e-8*np.eye(m)
            self.W = np.linalg.solve(A, K_nm.T@Y)
            self.alpha = None

    def decision_function(self, X):
        if self.X_anchors is None:
            return self.kernel_fn(X, self.X_train) @ self.alpha
        return self.kernel_fn(X, self.X_anchors) @ self.W

    def predict(self, X):
        return self.classes[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


#multiple kernel learning via KRR : combine kernels on different feature groups with learned or fixed weights.

class MKLRidgeClassifier:
    """
    Multiple Kernel Learning using KRR.

    Instead of one kernel on all features, we compute one kernel per feature
    group and combine them as a weighted sum:
        K_combined = w_hog * K_hog + w_lbp * K_lbp + w_color * K_color

    Each kernel uses the optimal gamma for its own feature group
    (computed as 1/(D_group * var_group)), so no single gamma needs to
    work well for all features simultaneously.

    This is the single most impactful improvement over plain KRR.
    """

    def __init__(self, lam=1.0, kernel_fns=None, weights=None, nystrom_m=None, seed=42):
        self.lam        = lam
        self.kernel_fns = kernel_fns   # dict {name: fn}
        self.weights    = weights      # dict {name: float} or None
        self.nystrom_m  = nystrom_m
        self.seed       = seed
        self.X_trains   = {}           # one X_train per group
        self.X_anchors  = {}           # Nystrom anchors per group
        self.W          = None
        self.alpha      = None
        self.classes    = None

    def _build_kernel(self, groups_A, groups_B, use_anchors=False):
        """
        Build combined kernel matrix from feature groups.
        """
        names = list(self.kernel_fns.keys())
        w = self.weights if self.weights else {n: 1.0/len(names) for n in names}
        K_sum = None
        for name in names:
            A = groups_A[name]
            if use_anchors and name in self.X_anchors:
                B = self.X_anchors[name]
            else:
                B = groups_B[name]
            K = self.kernel_fns[name](A, B)
            K_sum = w[name]*K if K_sum is None else K_sum + w[name]*K
        return K_sum

    def fit(self, groups, y):
        """groups: dict {name: (n, D)} normalised feature arrays."""
        self.classes = np.unique(y)
        n, K_cls = list(groups.values())[0].shape[0], len(self.classes)
        Y = np.zeros((n, K_cls), dtype=np.float64)
        for i, cls in enumerate(self.classes):
            Y[y==cls, i] = 1.0

        # Store training data references
        for name in self.kernel_fns:
            self.X_trains[name] = groups[name]

        if self.nystrom_m is None or self.nystrom_m >= n:
            print(f"  MKL-KRR full: building combined {n}x{n} kernel...")
            K = self._build_kernel(groups, groups)
            self.alpha = np.linalg.solve(K + self.lam*np.eye(n), Y)

        else:
            m = self.nystrom_m
            rng = np.random.RandomState(self.seed)
            anchor_idx = rng.choice(n, m, replace=False)
            # Same anchors for all groups (same subset of training points)
            anchors = {name: groups[name][anchor_idx] for name in self.kernel_fns}
            self.X_anchors = anchors

            print(f"  MKL-KRR Nystrom m={m}: building combined {n}x{m} kernel...")
            K_nm = self._build_kernel(groups, anchors, use_anchors=True) 

            K_mm = self._build_kernel(anchors, anchors).astype(np.float64)
            K_mm += 1e-8 * np.eye(m)
            A = self.lam*K_mm + K_nm.T@K_nm + 1e-8*np.eye(m)
            self.W = np.linalg.solve(A, K_nm.T@Y)
            self.alpha = None

    def decision_function(self, groups_test):
        if not self.X_anchors:
            K = self._build_kernel(groups_test, self.X_trains)
            return K @ self.alpha
        else:
            K_tm = self._build_kernel(groups_test, self.X_anchors, use_anchors=True)
            return K_tm @ self.W

    def predict(self, groups_test):
        return self.classes[np.argmax(self.decision_function(groups_test), axis=1)]

    def score(self, groups_test, y):
        return np.mean(self.predict(groups_test) == y)
