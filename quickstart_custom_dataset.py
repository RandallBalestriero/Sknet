from sknet import dataset
from sknet.utils import plotting
import numpy as np

# Load the custom dataset with any preferred method
# ...
# ...
# leading to (at least) the following loaded variables
# train_set, n_classes and data_format

train_set   = [np.random.randn(10000,1,32,32),
        np.random.randint(0,2,10000)]
n_classes   = 2
data_format = 'NCHW'

# any additional kwarg of the dataset should also be included
# create some random extra dataset features

noise_levels   = np.random.rand(10000)
missing_values = np.random.randn(10000,1,32,32)>0

my_dataset = dataset.custom(train_set=train_set,n_classes=n_classes,
                    data_format=data_format,noise_lebels=noise_levels,
                    missing_values=missing_values)

# it is also possible to add other keys after the initialization

classes               = ["name0","name1"]
my_dataset["classes"] = classes

