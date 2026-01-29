# Statistics:

'''

To compute stats of datasets we will use scipy module. SciPy is a scientific computation library that uses 
NumPy underneath. It stands for Scientific Python. It provides more utility functions for optimization, stats and 
signal processing. Like NumPy, SciPy is open source so we can use it freely.

'''

# Constants:

'''

As SciPy is more focused on scientific implementations, it provides many built-in scientific constants.
These constants can be helpful when you are working with Data Science. A list of all units under the constants module can 
be seen using the dir(constants) function.

Example:

from scipy import constants

print(constants.G)

'''

#  Constant Unit Categories:

'''

1. Metric (SI) Prefixes:

from scipy import constants

print(constants.yotta)    #1e+24
print(constants.zetta)    #1e+21
print(constants.exa)      #1e+18
print(constants.peta)     #1000000000000000.0
print(constants.tera)     #1000000000000.0
print(constants.giga)     #1000000000.0
print(constants.mega)     #1000000.0
print(constants.kilo)     #1000.0
print(constants.hecto)    #100.0
print(constants.deka)     #10.0
print(constants.deci)     #0.1
print(constants.centi)    #0.01
print(constants.milli)    #0.001
print(constants.micro)    #1e-06
print(constants.nano)     #1e-09
print(constants.pico)     #1e-12
print(constants.femto)    #1e-15
print(constants.atto)     #1e-18
print(constants.zepto)    #1e-21

2. Binary Prefixes:

from scipy import constants

print(constants.kibi)    #1024
print(constants.mebi)    #1048576
print(constants.gibi)    #1073741824
print(constants.tebi)    #1099511627776
print(constants.pebi)    #1125899906842624
print(constants.exbi)    #1152921504606846976
print(constants.zebi)    #1180591620717411303424
print(constants.yobi)    #1208925819614629174706176

3. Mass:

from scipy import constants

print(constants.gram)        #0.001
print(constants.metric_ton)  #1000.0
print(constants.grain)       #6.479891e-05
print(constants.lb)          #0.45359236999999997
print(constants.pound)       #0.45359236999999997
print(constants.oz)          #0.028349523124999998
print(constants.ounce)       #0.028349523124999998
print(constants.stone)       #6.3502931799999995
print(constants.long_ton)    #1016.0469088
print(constants.short_ton)   #907.1847399999999
print(constants.troy_ounce)  #0.031103476799999998
print(constants.troy_pound)  #0.37324172159999996
print(constants.carat)       #0.0002
print(constants.atomic_mass) #1.66053904e-27
print(constants.m_u)         #1.66053904e-27
print(constants.u) 

4. Angle:

from scipy import constants

print(constants.degree)     #0.017453292519943295
print(constants.arcmin)     #0.0002908882086657216
print(constants.arcminute)  #0.0002908882086657216
print(constants.arcsec)     #4.84813681109536e-06
print(constants.arcsecond)  #4.84813681109536e-06

5. Time:

from scipy import constants

print(constants.minute)      #60.0
print(constants.hour)        #3600.0
print(constants.day)         #86400.0
print(constants.week)        #604800.0
print(constants.year)        #31536000.0
print(constants.Julian_year) #31557600.0

6. Length:

from scipy import constants

print(constants.inch)              #0.0254
print(constants.foot)              #0.30479999999999996
print(constants.yard)              #0.9143999999999999
print(constants.mile)              #1609.3439999999998
print(constants.mil)               #2.5399999999999997e-05
print(constants.pt)                #0.00035277777777777776
print(constants.point)             #0.00035277777777777776
print(constants.survey_foot)       #0.3048006096012192
print(constants.survey_mile)       #1609.3472186944373
print(constants.nautical_mile)     #1852.0
print(constants.fermi)             #1e-15
print(constants.angstrom)          #1e-10
print(constants.micron)            #1e-06
print(constants.au)                #149597870691.0
print(constants.astronomical_unit) #149597870691.0
print(constants.light_year)        #9460730472580800.0
print(constants.parsec)            #3.0856775813057292e+16

7. Pressure:

from scipy import constants

print(constants.atm)         #101325.0
print(constants.atmosphere)  #101325.0
print(constants.bar)         #100000.0
print(constants.torr)        #133.32236842105263
print(constants.mmHg)        #133.32236842105263
print(constants.psi)         #6894.757293168361

8. Area:

from scipy import constants

print(constants.hectare) #10000.0
print(constants.acre)    #4046.8564223999992

9. Volume:

from scipy import constants

print(constants.liter)            #0.001
print(constants.litre)            #0.001
print(constants.gallon)           #0.0037854117839999997
print(constants.gallon_US)        #0.0037854117839999997
print(constants.gallon_imp)       #0.00454609
print(constants.fluid_ounce)      #2.9573529562499998e-05
print(constants.fluid_ounce_US)   #2.9573529562499998e-05
print(constants.fluid_ounce_imp)  #2.84130625e-05
print(constants.barrel)           #0.15898729492799998
print(constants.bbl)              #0.15898729492799998

10. Speed:

from scipy import constants

print(constants.kmh)            #0.2777777777777778
print(constants.mph)            #0.44703999999999994
print(constants.mach)           #340.5
print(constants.speed_of_sound) #340.5
print(constants.knot)           #0.5144444444444445

11. Temperature:

from scipy import constants

print(constants.zero_Celsius)      #273.15
print(constants.degree_Fahrenheit) #0.5555555555555556

12. Energy:

from scipy import constants

print(constants.eV)            #1.6021766208e-19
print(constants.electron_volt) #1.6021766208e-19
print(constants.calorie)       #4.184
print(constants.calorie_th)    #4.184
print(constants.calorie_IT)    #4.1868
print(constants.erg)           #1e-07
print(constants.Btu)           #1055.05585262
print(constants.Btu_IT)        #1055.05585262
print(constants.Btu_th)        #1054.3502644888888
print(constants.ton_TNT)       #4184000000.0

13. Power:

from scipy import constants

print(constants.hp)         #745.6998715822701
print(constants.horsepower) #745.6998715822701

14. Force:

from scipy import constants

print(constants.dyn)             #1e-05
print(constants.dyne)            #1e-05
print(constants.lbf)             #4.4482216152605
print(constants.pound_force)     #4.4482216152605
print(constants.kgf)             #9.80665
print(constants.kilogram_force)  #9.80665

'''

# Optimizers:

'''

Optimizers are a set of procedures defined in SciPy that either find the minimum value of a function, or the root of an equation.
Essentially, all of the algorithms in Machine Learning are nothing more than a complex equation that needs to be minimized with the 
help of given data.

'''

# Roots of an Equation:

'''

NumPy is capable of finding roots for polynomials and linear equations, but it can not find roots for non linear equations, 
like this one: x + cos(x)

For that you can use SciPy's optimize.root function. This function takes two required arguments:

variable = root(func, initial value)
print(variable.x) : stores that value of x at which root was found.

Example:

from scipy.optimize import root
from math import cos

def eqn(x):
  return x + cos(x)

myroot = root(eqn, 0)

print(myroot.x)

'''

# Finding Minima:

'''

We can use scipy.optimize.minimize() function to minimize the function. The minimize() function takes the following arguments:

variable(func, initial value)
print(variable.x) stores that value of x at which minimum was found.

method : name of the method to use Legal values:
    'CG'
    'BFGS'
    'Newton-CG'
    'L-BFGS-B'
    'TNC'
    'COBYLA'
    'SLSQP'

Example:

from scipy.optimize import minimize

def eqn(x):
  return x**2 + x + 2

mymin = minimize(eqn, 0, method='BFGS')

print(mymin.x)

'''

# Types of Arrays:

'''

Sparse Data: is a data set where most of the item values are zero. 
Dense Array: has most of the values that are not zero.

In scientific computing, when we are dealing with partial derivatives in linear algebra we will come across sparse data.
SciPy has a module, scipy.sparse that provides functions to deal with sparse data.
There are primarily two types of sparse matrices that we use:

1. CSC - Compressed Sparse Column. For efficient arithmetic and fast column slicing.
2. CSR - Compressed Sparse Row. For faster matrix vector products and fast row slicing, 

We will use the CSR matrix.

Example:

import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])

print(csr_matrix(arr))

This will return: 

Cords    Values
(0, 5)   1
(0, 6)   1
(0, 8)   2

Operations:

1. data() : Used to View stored data not the zero items.
2. count_nonzero() : Used to count nonzeros.
3. eliminate_zeros() : Used to eliminate zeros.
4. sum_duplicates() : Used to sum duplicates to eliminate them.
5. tocsc(): Used for converting csr to csc.

'''

# Statistical Description of Data:

'''

In order to see a summary of values in an array, we can use the describe() function.
It returns the following description:

1. number of observations (nobs)
2. minimum and maximum values = minmax
3. mean
4. variance
5. skewness
6. kurtosis

Example:

import numpy as np
from scipy.stats import describe

v = np.random.normal(size=500)

print(describe(v))

'''

# Statistical Significance:

'''

In statistics, statistical significance means that the result that was produced has a reason behind it, it was not produced randomly, 
or by chance. SciPy provides us with a module called scipy.stats, which has functions for performing statistical significance tests.

Here are some techniques and keywords that are important when performing such tests:

1. Hypothesis: Hypothesis is an assumption about a parameter in population. Tere are two types of hypothesis:

a. Null Hypothesis: It assumes that the observation is not statistically significant.
b. Alternate Hypothesis: It assumes that the observations are due to some reason.

There are two methods to test hypothesis:

1. One tailed test: When our hypothesis is testing for one side of the value only, it is called "one tailed test".

Example:

For the null hypothesis: "the mean is equal to k",
then for alternate hypothesis it will be: "the mean is less than k" or "the mean is greater than k".

2. Two tailed test: When our hypothesis is testing for both side of the values.

Example:

For the null hypothesis: "the mean is equal to k".
then for alternate hypothesis it will be: "the mean is not equal to k" so In this case the mean is less than, or greater than k, 
and both sides are to be checked.

Alpha value: Alpha value is the level of significance. It is usually taken as 0.01, 0.05, or 0.1.
P value: P value tells how close to extreme the data actually is. P value and alpha values are compared to establish the 
statistical significance. If p value <= alpha we reject the null hypothesis and say that the data is statistically significant. 
otherwise we accept the null hypothesis.

T-Test:
T-tests are used to determine if there is significant differnce between means of two variables and lets us know if they belong 
to the same distribution. It is a two tailed test.

The function ttest_ind() takes two samples of same size and produces a tuple of t-statistic and p-value.

Example:

import numpy as np
from scipy.stats import ttest_ind

v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)

print(ttest_ind(v1, v2))

KS-Test:
KS test is used to check if given values follow a distribution. It can be used as a one tailed or two tailed test.
print(kstest(variable, CDF))

A CDF can be either a string like normal or exponential or a callable function that returns the probability.

Example:

import numpy as np
from scipy.stats import kstest

v = np.random.normal(size=100)

print(kstest(v, 'norm'))

'''

# Normality Tests (Skewness and Kurtosis):

'''

Normality tests are based on the skewness and kurtosis. The normaltest() function returns p value for the null hypothesis.

Example:

import numpy as np
from scipy.stats import normaltest

v = np.random.normal(size=100)

print(normaltest(v))

Skewness: A measure of symmetry in data. For normal distributions it is 0.

If it is negative, it means the data is skewed left.
If it is positive it means the data is skewed right.

Kurtosis: A measure of weight of data.

Positive kurtosis means heavy tailed.
Negative kurtosis means lightly tailed.

Example (Finding skewness and kurtosis):

import numpy as np
from scipy.stats import skew, kurtosis

v = np.random.normal(size=100)

print(skew(v))
print(kurtosis(v))

'''
