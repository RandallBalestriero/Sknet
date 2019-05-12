from pylab import *
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Hyperplane:
    def __init__(self,slope,bias):
        self.slope = slope
        self.bias  = bias
        # assume the last component is not 0
        #create a function that takes points in
        #$R^{len(slope)-1}$
        print(shape(bias),shape(squeeze(slope)),bias)
        
    def hyperplane(self,x):
        if len(self.slope)==2:
            return -(self.bias+x*self.slope[0])/self.slope[1]
        else:
            return -(self.bias+dot(x,self.slope[:-1]))/self.slope[-1]
    def project(self,x):
        return dot(x,self.slope)+self.bias


class Layer:
    def __init__(self,W,b):
        self.K,self.D = W.shape
        self.hyperplanes = [Hyperplane(w_,b_) for w_,b_ in zip(W,b)]
#        print([h.slope for h in self.hyperplanes])
    def project(self,x):
        if(self.K>1):
            return stack([h.project(x) for h in self.hyperplanes],1)
        else:
            return array([h.project(x) for h in self.hyperplanes])


class Space:
    def __init__(self,MIN_X,MAX_X,MIN_Y,MAX_Y,N):
        self.x,self.y = meshgrid(linspace(MIN_X,MAX_X,N),linspace(MIN_Y,MAX_Y,N))
        self.X        = concatenate([self.x.reshape((-1,1)),self.y.reshape((-1,1))],axis=1)
 
 
def circle_2d(center, radius , N):
    """Given a center, a radius and a number of points, return a uniform of the
    the 2D circle 

    Parameters
    ----------

    center : numpy.array
        the center of the circle, must be a scalar of a vector of length 2.

    radius : float
        the radius of the circle, must be nonnegative

    N : int
        the number of points to be used for the circle discretization, must be
        nonnegative
    """
    t = np.arange(N)/N
    PI= 2*3.14159
    X = np.stack([RADIUS*np.cos(t*PI),RADIUS*np.sin(t*PI)],1)
    X+= CENTER
    return X






def get_input_space_partition(states,N,M,duplicate=1):
    #the following should take as input a collection of points
    #in the input space and return a list of binary states, each 
    #element of the list is for 1 specific layer and it is a 2D array
    if states.ndim>1:
        states = states2values(states)
    partitioning  = grad(states.reshape((N,M)).astype('float32'),duplicate)
    return partitioning




def grad(x,duplicate=0):
    #compute each directional (one step) derivative as a boolean mask
    #representing jump from one region to another and add them (boolean still)
    g_vertical   = np.greater(np.pad(np.abs(x[1:]-x[:-1]),((1,0),(0,0)),'constant'),0)
    g_horizontal = np.greater(np.pad(np.abs(x[:,1:]-x[:,:-1]),[[0,0],[1,0]],'constant'),0)
    g_diagonaldo = np.greater(np.pad(np.abs(x[1:,1:]-x[:-1,:-1]),[[1,0],[1,0]],'constant'),0)
    g_diagonalup = np.greater(np.pad(np.abs(x[:-1:,1:]-x[1:,:-1]),[[1,0],[1,0]],'constant'),0)
    overall      = g_vertical+g_horizontal+g_diagonaldo+g_diagonalup
    if duplicate>0:
        overall                 = np.stack([np.roll(overall,k,1) for k in range(duplicate+1)]\
                                    +[np.roll(overall,k,0) for k in range(duplicate+1)]).sum(0)
        overall[:duplicate]    *= 0
        overall[-duplicate:]   *= 0
        overall[:,:duplicate]  *= 0
        overall[:,-duplicate:] *= 0
    return np.greater(overall,0).astype('float32')





def states2values(states, state2value_dict=None, return_dict=False):
    """Given binary masks obtained for example from the ReLU activation, 
    convert the binary vectors to a single float scalar, the same scalar for
    the same masks. This allows to drastically reduce the memory overhead to
    keep track of a large number of masks. The mapping mask -> real is kept
    inside a dict and thus allow to go from one to another. Thus given a large
    collection of masks, one ends up with a large collection of scalars and
    a mapping mask <-> real only for unique masks and thus allow reduced memory
    requirements.

    Parameters
    ----------

    states : bool matrix
        the matrix of shape (#samples,#binary values). Thus if using a deep net
        the masks of ReLU for all the layers must first be flattened prior
        using this function.

    state2value_dict : dict
        optional dict containing an already built mask <-> real mapping which
        should be used and updated given the states value.

    return_dict : bool
        if the update/created dict should be return as part of the output

    Returns
    -------

    values : scalar vector
        a vector of length #samples in which each entry is the scalar
        representation of the corresponding mask from states

    state2value_dict : dict (optional)
        the newly created or updated dict mapping mask to value and vice-versa.


    """
    if state2value_dict is None:
        state2value_dict = dict()
    values = zeros(states.shape[0])
    for i,state in enumerate(states):
        str_s = str(state.astype('uint8')).replace(' ','')[1:-1]
        if(str_s not in state2value_dict):
            state2value_dict[str_s] = randn()
        values[i] = state2value_dict[str_s]
    return values

