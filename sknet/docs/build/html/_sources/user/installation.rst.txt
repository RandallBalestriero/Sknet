Installation
============

Requirements:
_Python 3. 
_Tensorflow set up with GPU
Set up environment variables DATASET_PATH and SAVE_PATH

An easy way to get everything installed nad set-up out-of-the-box is to
perform the installation via _Anaconda.
To do this, we propose a configuration file (not minimal) that can be 
used as is to set up an environment.

Install Conda::

   wget https://repo.anaconda.com/archive/Anaconda3-2018.12-MacOSX-x86_64.sh #(python3.)
   bash Anaconda-latest-Linux-x86_64.sh
   #Create environment with packages
   conda create --name tf3 --file ranet/config/conda_spec-file.txt
   # OR Install packages into current (myenv) environment
   conda install --name myenv --file conda_spec-file.txt



.. _Python: https://www.python.org/download/releases/3.0/
.. _Tensorflow: https://www.tensorflow.org/
.. _Anaconda: https://www.anaconda.com/
