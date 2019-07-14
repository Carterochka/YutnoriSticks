import numpy as np
import matplotlib.pyplot as plt
import math

### creating 100 angle values with step of 1 degree
x_o = range(0, 101)

x = []
for n in x_o:
    t = n*1.0/101 * math.pi/2
    x.append(t)

y = []

### calculating probability using formula from korean article
for n in x:
    t = math.atan( math.cos(n) / (math.sin(n) + 2*(math.cos(n)**3)
                                  / (3*(math.pi/2 + n + math.sin(n)*math.cos(n))) ) ) / math.pi
    y.append(t)

### plotting values calculated by formula
plt.plot(x, y, color='blue', label='Formula from article')


### calculating line equation for one that connects first and last points
xl1 = x[0]
xl2 = x[len(x) - 1]
yl1 = y[0]
yl2 = y[len(y) - 1]

k = (yl2 - yl1)/(xl2 - xl1)
b = yl1 - k*xl1
yl = []

for n in x:
    t = k*n + b
    yl.append(t)

### plotting this line
plt.plot(x, yl, color='red', label='Line between first and last dots')
print('Line between first and last points:')
print('k = %.4f' % k)
print('b = %.4f' % b)

i = 0
max = 0
while i < len(y):
    if y[i] - yl[i] > max:
        max = y[i] - yl[i]
    i += 1

print('max distance: %.4f' % max)


### creating linear regression for this set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

xnp = np.array(x).reshape(-1,1)
reg.fit(xnp,y)
print('Linear regression: ')
print('k = %.4f' % reg.coef_[0])
print('b = %.4f' % reg.intercept_)
print('r-squared: %.4f' % reg.score(xnp, y))

yr = []
for n in x:
    t = reg.intercept_ + reg.coef_[0] * n
    yr.append(t)

plt.plot(x, yr, color='#9900FF', label='Linear regression')

plt.legend()
plt.show()
