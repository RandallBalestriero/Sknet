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
 

class CircleSpace:
    def __init__(self,CENTER,RADIUS,N):
        x,y    = meshgrid(linspace(CENTER-RADIUS,CENTER+RADIUS,N),linspace(CENTER-RADIUS,CENTER+RADIUS,N))
        mask   = sqrt(x**2+y**2)<=RADIUS
        self.x = x[mask]
        self.y = y[mask]
        self.X = concatenate([self.x.reshape((-1,1)),self.y.reshape((-1,1))],axis=1)
 
class Circle:
    def __init__(self,CENTER,RADIUS,N):
        x = linspace(-RADIUS,RADIUS,N)
        y = sqrt(RADIUS**2-x**2)
        self.X = concatenate([
            concatenate([x.reshape((-1,1)),y.reshape((-1,1))],axis=1),
            concatenate([x[::-1].reshape((-1,1)),-y.reshape((-1,1))],axis=1)],0)
        self.X+=CENTER.reshape((1,-1))





















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





def states2values(states):
    #get an array of binary values representing the relu 
    #or absolute value etc state and thus is 1 if the relu
    #was active and 0 otherwise and this for each unit
    #so states is a 2D array of shape (#samples,#units)
    state2values_dict = dict()
    values            = zeros(states.shape[0])
    for i in range(states.shape[0]):
        str_s = str(states[i].astype('uint8'))
        if(str_s not in state2values_dict):
            state2values_dict[str_s] = randn()
        values[i]=state2values_dict[str_s]
    return values




