import numpy as np
from Functions import VBA_basics as base

Q = np.eye(5)*2
Q[1, 1] = np.nan
Q = np.array([[1, 3, 5],[4, 2, 5],[1, 2, 3]])

print(base.log_det(Q))

