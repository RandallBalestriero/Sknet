from pylab import *


import utils as ut


# test negative dimension and 0
print("###########\nTesting Dimensions of Space class\n###########")
try: space=ut.Space(-1)
except Exception as e: print(e)

try: space=ut.Space(0)
except Exception as e: print(e)


# test wrong name
print("\n\n\n\n\n#############\nTesting name of Space class\n############")
space=ut.Space(4,axes_name=['f'])
space=ut.Space(4)

## test hyperplane
#print("\n\n\n\n\n#############\nTesting hyperplane class\n############")
#hyperplane=ut.Hyperplane(space,randn(4))






# test intersection
space=ut.Space(2)
h1 = ut.Hyperplane(space,array([1,1]),-1)
h2 = ut.Hyperplane(space,zeros(2),0)

print(ut.intersect(h1,h2))



