import numpy as np

class SVM:
    def __init__(self, X, y, C=1, kernel='linear', b=0, max_iter=300, tol=1e-5, eps=1e-8):
        self.X = X
        self.y = y
        self.m, self.n = np.shape(self.X)
        self.C = C

        self.alphas = np.zeros(self.m)
        self.b = b

        self.kernel = kernel       # 'linear', 'rbf'
        if kernel == 'linear':
            self.kernel_func = self.linear_kernel
        elif kernel == 'gaussian' or kernel == 'rbf':
            self.kernel_func = self.gaussian_kernel
        else:
            raise ValueError('unknown kernel type')

        self.error = np.zeros(self.m)

        self.max_iter=max_iter
        self.tol = tol
        self.eps = eps


        self.is_linear_kernel = True if self.kernel == 'linear' else False
        self.w = np.zeros(self.n)  # used by linear kernel
    
    def linear_kernel(self, x1, x2, b=0):
        return x1 @ x2.T + b
    
    def gaussian_kernel(self, x1, x2, sigma=1):
        if np.ndim(x1) == 1 and np.ndim(x2) == 1:
            return np.exp(-(np.linalg.norm(x1-x2,2))**2/(2*sigma**2))
        elif(np.ndim(x1)>1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2)>1):
            return np.exp(-(np.linalg.norm(x1-x2, 2, axis=1)**2)/(2*sigma**2))
        elif np.ndim(x1) > 1 and np.ndim(x2) > 1 :
            return np.exp(-(np.linalg.norm(x1[:, np.newaxis] \
                             - x2[np.newaxis, :], 2, axis = 2) ** 2)/(2*sigma**2))
        return 0.
    
    def predict(self, x):
        result = (self.alphas * self.y) @ self.kernel_func(self.X, x) + self.b
        return result

    def get_error(self, i):
        return self.predict(self.X[i,:]) - self.y[i]

    def take_step(self, i1, i2):
        if (i1 == i2):
            return 0

        x1 = self.X[i1, :]
        x2 = self.X[i2, :]

        y1 = self.y[i1]
        y2 = self.y[i2]

        alpha1 = self.alphas[i1]
        alpha2 = self.alphas[i2]

        b = self.b

        E1 = self.get_error(i1)
        E2 = self.get_error(i2)

        s = y1 * y2

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)

        if L == H:
            return 0

        k11 = self.kernel_func(x1, x1)
        k12 = self.kernel_func(x1, x2)
        k22 = self.kernel_func(x2, x2)

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:
            # Abnormal case for eta <= 0, treat this scenario as no progress
            return 0

        # Numerical tolerance
        # if abs(alpha2_new - alpha2) < self.eps:   # this is slower
        # below is faster, not degrade the SVM performance
        if abs(alpha2_new - alpha2) < self.eps * (alpha2 + alpha2_new + self.eps):
            return 0

        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)

        # Numerical tolerance
        if alpha1_new < self.eps:
            alpha1_new = 0
        elif alpha1_new > (self.C - self.eps):
            alpha1_new = self.C

        # Update threshold
        b1 = b - E1 - y1 * (alpha1_new - alpha1) * k11 - y2 * (alpha2_new - alpha2) * k12
        b2 = b - E2 - y1 * (alpha1_new - alpha1) * k12 - y2 * (alpha2_new - alpha2) * k22
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = 0.5 * (b1 + b2)

        # Update weight vector for linear SVM
        if self.is_linear_kernel:
            self.w = self.w + y1 * (alpha1_new - alpha1) * x1 \
                            + y2 * (alpha2_new - alpha2) * x2

        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        # Error cache update
        ## if alpha1 & alpha2 are not at bounds, the error will be 0
        self.error[i1] = 0
        self.error[i2] = 0

        i_list = [idx for idx, alpha in enumerate(self.alphas) \
                      if 0 < alpha and alpha < self.C]
        for i in i_list:
            self.error[i] += \
                  y1 * (alpha1_new - alpha1) * self.kernel_func(x1, self.X[i,:]) \
                + y2 * (alpha2_new - alpha2) * self.kernel_func(x2, self.X[i,:]) \
                + (self.b - b)

        return 1


    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.get_error(i2)
        r2 = E2 * y2

        # Choose the one that is likely to violiate KKT
        # if (0 < alpha2 < self.C) or (abs(r2) > self.tol):  # this is slow
        # below is faster, not degrade the SVM performance
        if ((r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0)):
            if len(self.alphas[(0 < self.alphas) & (self.alphas < self.C)]) > 1:
                if E2 > 0:
                    i1 = np.argmin(self.error)
                else:
                    i1 = np.argmax(self.error)

                if self.take_step(i1, i2):
                    return 1

            # loop over all non-zero and non-C alpha, starting at a random point
            i1_list = [idx for idx, alpha in enumerate(self.alphas) \
                           if 0 < alpha and alpha < self.C]
            i1_list = np.roll(i1_list, np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

            # loop over all possible i1, starting at a random point
            i1_list = np.roll(np.arange(self.m), np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

        return 0
    
    def fit(self):
        loop_num = 0
        numChanged = 0
        examineAll = True
        while numChanged > 0 or examineAll:
            if loop_num >= self.max_iter:
                break

            numChanged = 0
            if examineAll:
                for i2 in range(self.m):
                    numChanged += self.examine_example(i2)
            else:
                i2_list = [idx for idx, alpha in enumerate(self.alphas) \
                                if 0 < alpha and alpha < self.C]
                for i2 in i2_list:
                    numChanged += self.examine_example(i2)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

            loop_num += 1
