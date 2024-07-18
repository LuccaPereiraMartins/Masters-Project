# Second numerical example

import math
import numpy as np
import matplotlib.pyplot as plt
from decimal import *

getcontext().prec = 25

def taylorseries(x,d):
    return Decimal(sum([((-1)**i)*((x**((2*i)+1))/math.factorial((2*i)+1)) for i in range(d)]))

fidelities = 5
xtest = np.linspace(0,2*math.pi,100)


for fid in range(fidelities):
    plt.plot(xtest,[taylorseries(x,fid+5) for x in xtest])

plt.plot(xtest,[math.sin(x) for x in xtest])
plt.show()

# Not best function because of how inaccurate it is for large values of x