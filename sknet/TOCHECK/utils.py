import warnings
import pylab as pl
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr as sparse_lsqr

def set_minus(list1,list2):
    return [l for l in list1 if l not in list2]


class SparseVector:
    def __init__(self,indexes,values,length):
        if len(indexes)!=len(values): raise ValueError(
                "SpaceVector: length of indexes != length of values\
                        ({}!={})".format(len(indexes),len(values)))
        self.length  = length
        self.indexes = indexes
        self.values  = values

class Space(object):
    def __init__(self,dimension,name="",axes_name=[]):
        """
        dimension: (positive integer) describes the dimension of the space.
        name     : (str) name of the space
        axes_name: (list of str) the name of each dimension. 
                    Must be same length as dimension."""
        #Checking Arguments
        if dimension<=0: raise ValueError("dimension must be positive")
        if type(name) is  not str: raise TypeError("name must be a str")
        self.dimension = dimension
        # Generate the axes names
        if(len(axes_name)==0): 
            warnings.warn("Space class init: No axes_names given,\
                            using default (x_1,...)")
            self.axes_name = pl.array(["x_{}".format(d+1) for d in range(dimension)])
        elif(len(axes_name)<dimension): 
            warnings.warn("Space class init: len(axes_name)<dimension,\
                    using default (x_1,...)")
            self.axes_name = pl.array(["x_{}".format(d+1) for d in range(dimension)])
        else: self.axes_name=axes_name
        # Generate the space name
        if(name is ""):
            warnings.warn("Space class init: No name given, using default R^K")
            self.name = "R^{}".format(dimension)
        else: self.name = name
        # Printing
        print("Initialization of Space:\n\
                \tName : {}\n\tDimensions : {}\n\
                \tAxes name : {}".format(self.name,self.dimension,self.axes_name))




class Hyperplane(object):
    def __init__(self,space,w,b=0.,name=""):
        """
        ------
        This class defines a linear projection f(x) = <w,x>+b as an algebraic
        object to compute basis, nullspace and various operations/properties
        ------
        space: (Space instance) used to port the axes_names and the dimension
        w    : (list or array) must be of length space.dimension. Defines the 
                linear hyperplane coefficients
        b    : (real or vector) the bias of the hyperplane. If a vector it
                represent the direction in which to push the linear hyperplane\
                        of coefficients w. If a scalar then it is assumed that\
                        the bias vector is made of 0 and b for the last dimension
        ------"""
        self.ambiant_dimension = space.dimension+1
        self.input_dimension   = space.dimension
        self.dim               = space.dimension
        self.name              = name if name is not "" else "hyperplane"
        # Setting bias
        if pl.isscalar(b): 
            self.bias     = pl.zeros(self.ambiant_dimension)
            self.bias[-1] = b
        else:
            if len(b)<self.input_dimension: raise ValueError("Hyperplane class\
                    init: given len(b)<ambiant dimension\
                    ({}<{})".format(len(b),self.ambiant_dimension))
            self.biasias  = b
        # Setting the slope
        if len(w)<self.input_dimension: raise ValueError(
                "Hyperplane class init: given len(w)<\
                ambiant dimension ({}<{})".format(len(w),self.input_dimension))
        self.slope = w
        # Generate current basis
        # it is a sparse matrix of shape (ambiant_dim,dim)
        self.basis = self._generate_basis()
        print("Initialization of Hyperplane:\n\
                \tName : {}\n\
                \tAmbiant Dimension : {}\n\
                \tInput Dimension : {}\n\
                \tw : {}\n\
                \tb : {}".format(self.name,self.ambiant_dimension,\
                self.input_dimension,self.slope,self.bias))
    def _generate_basis(self):
        """
        This method generates a basis of the linear manifold w.r.t the \
        canonical basis. Each basis vector leaves in the ambiant space \
        dimension and the number of vectors is equal to intrinsic dimension.
        """
        data    = pl.stack([pl.ones(self.input_dimension),
                  self.slope],1).flatten().astype('float32')
        indices = pl.stack([pl.arange(self.input_dimension),
                  pl.full(self.input_dimension,self.input_dimension)],1).flatten().astype('int32')
        indptr  = pl.full(self.input_dimension+1,2,dtype=pl.float32)
        indptr[0] = 0
        basis   = sp.csc_matrix((data,indices,pl.cumsum(indptr)),
                shape=(self.ambiant_dimension,self.input_dimension))
        return basis
    def _generate_basis2(self):
        """
        This method generates a basis of the linear manifold w.r.t the canonical basis
        Each basis vector leaves in the ambiant space dimension and the number
        of vectors is equal to intrinsic dimension.
        """
        #First get the canonical basis that are not needed to span the linear manifold
        nonzero_indexes   = pl.nonzero(self.slope) # this must be of length self.intrinsic_dimension
        assert(len(nonzero_indexes)==self.intrinsic_dimension)
        # now pick one of the dimension for the denominator, pick the biggest for stability
        denominator_index = abs(self.slope).argmax()
        denominator_value = self.slope[denominator_index]
        basis             = [SparseVector([nonzero_indexes[i],self.ambiant_dimension-1],[1,-self.slope[nonzero_indexes[i]]],self.ambiant_dimension) for i in set_minus(nonzero_indexes,[denominator_index])]
        return basis



def intersect(h1,h2):
    if h1.ambiant_dimension!=h2.ambiant_dimension:
        raise ValueError("Different ambiant\
            spaces dimensions ({}!={})".format(len(w1),len(w2)))
#    elif pl.allclose(w1,w2) and pl.allclose(b1,b2):
#        print("return same object")
#        return copy(h1)
#    elif pl.allclose(w1,w2) and not pl.allclose(b1,b2):
#        print("return empty set")
#        return None
     
    #Now that the trivial cases have been removed there exists an 
    #intersection of the two hyperplanes to obtain the new form of
    #the hyperplane we perform the following 3 steps:
    # (1)consider the intersection of two affine spaces as the intersection
    #    of a linear space and an affine space with modified bias
    # (2)express the bias of the affine space in the linear space,
    #    this gives us the bias that will be for the new affine space
    # (3)compute the basis of the intersection of the two linear spaces
    #    this gives us the span of the linear space, jointly with the above
    #    bias corresponding to the affine space from the intersection 
    #(1)
    # we set h2 to be without bias and thus we alter h1 bias as follows
    bp     = h1.bias-h2.bias
    #(2)
    A      = sp.hstack([h1.basis,h2.basis])
    output = sparse_lsqr(A,bp)
    alpha_beta_star,istop,itn=output[:3]
    if(istop==2):
        warnings.warn("In Intersection of {} with {},\
                least square solution is approximate".format(h1.name,h2.name))
    # since we set h2 to be without bias we need to express the bias vector w.r.t.
    # the h2 basis as follows
    bpp    = h2.basis.dot(alpha_beta_star[h1.dim:])
    print("New bias is {}".format(bpp))
    # (3) --------- Implement the Zassenhaus algorithm -------------
    # assumes each space basis is in the same basis
    # first need to create the original matrix
    print(h1.basis.todense(),'\n\n')
    print(h2.basis.todense(),'\n\n')
    matrix = sp.bmat([[h1.basis,h2.basis],[h1.basis,None]]).T.todense()
    # Now we need to put the top left half in row echelon form
    # loop over the columns of the left half
    rows = range(matrix.shape[0])
    for column in range(h1.dim+h2.dim):
#        print('start column:{}\nmatrix:\n{}\n\n'.format(column,matrix))
        # compute current index and value of pivot A_{column,column}
        pivot           = matrix[rows[column],column]
        # check for active pivoting to switch current row 
        # with the one with maximum value
        maximum_index   = matrix[rows[column:],column].argmax()+column
        if maximum_index>column:
            print('needs preventive partial pivoting')
            rows[column]        = maximum_index
            rows[maximum_index] = column
            pivot               = matrix[rows[maximum_index],column]
        # Loop over the rows
        multiplicators = matrix[rows[column+1:],column]/pivot
        matrix[rows[column+1:]]-=multiplicators.reshape((-1,1))*matrix[rows[column]].reshape((1,-1))
        if(pl.allclose(matrix[rows[column+1:]],0)): return matrix












def intersect_sparse(h1,h2):
    if h1.ambiant_dimension!=h2.ambiant_dimension:
        raise ValueError("Different ambiant\
            spaces dimensions ({}!={})".format(len(w1),len(w2)))
#    elif pl.allclose(w1,w2) and pl.allclose(b1,b2):
#        print("return same object")
#        return copy(h1)
#    elif pl.allclose(w1,w2) and not pl.allclose(b1,b2):
#        print("return empty set")
#        return None
     
    #Now that the trivial cases have been removed there exists an 
    #intersection of the two hyperplanes to obtain the new form of
    #the hyperplane we perform the following 3 steps:
    # (1)consider the intersection of two affine spaces as the intersection
    #    of a linear space and an affine space with modified bias
    # (2)express the bias of the affine space in the linear space,
    #    this gives us the bias that will be for the new affine space
    # (3)compute the basis of the intersection of the two linear spaces
    #    this gives us the span of the linear space, jointly with the above
    #    bias corresponding to the affine space from the intersection 
    #(1)
    bp     = h2.bias-h1.bias
    #(2)
    A      = sp.hstack([h1.basis,h2.basis])
    output = sparse_lsqr(A,bp)
    alpha_beta_star,istop,itn=output[:3]
    if(istop==2):
        warnings.warn("In Intersection of {} with {},\
                least square solution is approximate".format(h1.name,h2.name))
    bpp    = h2.basis.dot(alpha_beta_star[h1.dim:])
    print("New bias is {}".format(bpp))
    # (3) --------- Implement the Zassenhaus algorithm -------------
    # assumes each space basis is in the same basis
    # first need to create the original matrix
    print(h1.basis.todense(),'\n\n')
    print(h2.basis.todense(),'\n\n')
    matrix = sp.bmat([[h1.basis,h2.basis],[h1.basis,None]]).T
    # Now we need to put the top left half in row echelon form
    # loop over the columns of the left half
    for column in range(h1.ambiant_dimension-1):
        print('start column:{}\nmatrix:\n{}\n\n'.format(column,matrix.todense()))
        # compute current index and value of pivot A_{column,column}
        pivot_index     = pl.flatnonzero((matrix.col==column) & (matrix.row == column))
        pivot           = pl.squeeze(matrix.data[pivot_index])
        # compute indices to act upon
        current_indices = pl.flatnonzero((matrix.col==column) & (matrix.row > column))
        current_data    = matrix.data[current_indices]
        # check for active pivoting to switch current row 
        # with the one with maximum value
        maximum_index   = current_indices[current_data.argmax()]
        maximum_value   = matrix.data[maximum_index]
        print('maximum value',maximum_value,'pivot',pivot)
        if abs(maximum_value)>abs(pivot):
            print('needs preventive pivoting')
            row_w_maximum                         = matrix.row[maximum_index]
            matrix.row[matrix.row==row_w_maximum] = column
            matrix.row[matrix.row==column]        = row_w_maximum
            pivot                                 = pl.squeeze(maximum_value)
            print('\tstart column:{}\n\tmatrix:\n\t{}\n\n'.format(column,matrix.todense()))
        # create a buffer to add new values
        to_add = {'row':[],'col':[],'data':[]}
        considered_column_indices = pl.flatnonzero((matrix.col>=column) & (matrix.row == column))
        # Loop over the rows
        for row,value in zip(matrix.row[current_indices],matrix.data[current_indices]):
            multiplicator = value/pivot
            print('Multiplicator={}'.format(multiplicator))
            # Compute which column of the current row are nonzeros
            pivot_nonzero_indices = pl.flatnonzero((matrix.col>=column) & (matrix.row == row))
            # Find the columns that are both nonzeros at row column and row row
            both_nonzero = pl.array([i for i in pivot_nonzero_indices if i in considered_column_indices])
            print(both_nonzero)
            # Find here zero but pivot row nonzero, this corresponds to the column that will
            # need to have new values in it
            hereonly_nonzero = pl.array([i for i in considered_column_indices if i not in pivot_nonzero_indices])
            # Loop over the columns which are both nonzeros
            if(len(both_nonzero)>0):
                for pos,val in zip(both_nonzero,pl.nditer(matrix.data[both_nonzero],op_flags=['readwrite'])):
                    val-=multiplicator*value
                    print('New Value',val)
            # Loop over column that are here zero and thus need a new placeholder
            if(len(hereonly_nonzero)>0):
                for pos in hereonly_nonzero:
                    to_add['row'].append(row)
                    to_add['data'].append(-value)
                    to_add['col'].append(pos)
        print(matrix.todense())
        # add the newly introduces nonzero values ot the sparse matrix
        matrix.row = pl.concatenate([matrix.row,to_add['row']])
        matrix.col = pl.concatenate([matrix.col,to_add['col']])
        matrix.data = pl.concatenate([matrix.data,to_add['data']])
        print(matrix.todense())
        matrix.eliminate_zeros()
#        # now need to remove the entries that are now 0
#        # first from the ones explicitly reduces
#        to_delete        = pl.flatnonzero((matrix.col==column) & (matrix.row > column))
#        # now form the possible ones that became 0 during row operations
#        indices          = pl.flatnonzero(matrix.row<column)
#        data             = matrix.data[indices]
#        extras_to_delete = pl.flatnonzero(pl.isclose(data,0))
#        # merge both lists
#        to_delete        = pl.concatenate([to_delete,extras_to_delete])
#        # delete the corresponding indices alternative: matrix.eliminate_zeros()
#        matrix.col  = pl.delete(matrix.col,to_delete, None)
#        matrix.row  = pl.delete(matrix.row,to_delete, None)
#        matrix.data = pl.delete(matrix.data,to_delete, None)
#        # end of this procedure, move over to the next column
#        # check if completed
        block_indices = pl.flatnonzero((matrix.col<h1.ambiant_dimension) & (matrix.row > column))
        if(len(current_indices)==0): return matrix.to_dense()


























