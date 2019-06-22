from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def create_dataset(hm, variance, step=2, correlation=False):        # hm = number of points(how much) ,
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':                   # positive corelation will add to value of ys
            val += step
        elif correlation and correlation == 'neg':                 # negative correlation will reduce the value of ys
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):                       # using mean square method for calculating the slope.
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - m * mean(xs)                                 # defining b based on mean values

    return m, b


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    print(squared_error_regr)
    print(squared_error_y_mean)

    r_squared = 1 - (squared_error_regr / squared_error_y_mean)        # r is coefficient correlation

    return r_squared



#xs = np.array([1,2,3,4,5], dtype=np.float64)
#ys = np.array([5,4,6,5,6], dtype=np.float64)

xs, ys = create_dataset(40, 10, 2, correlation='pos')       # with smaller variance we have better corelation coefficient

m, b = best_fit_slope_and_intercept(xs, ys)                    # correlation can be 'pos' or 'neg' which determine the positivity or negativity od line slop
regression_line = [(m * x) + b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys, color='#003F72', label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
