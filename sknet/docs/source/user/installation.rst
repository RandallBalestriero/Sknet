Installation
============

.. rubric:: Requirements
    - _Python 3. 
    - _CuDNN
    - _Tensorflow set up with GPU
    - (h5py, numpy, ...)

.. rubric:: Environment variables

Set up the environment variable DATASET_PATH to where you want sknet
to put (when downloaded) all the dataset by default. It should be a valid path.
It can be added in yoyr :file:`.bashrc` as 
``export DATASET_PATH='/home/user/DATASET/'``
 

.. rubric:: Installation (Anaconda)

An easy way to get everything installed and set-up out-of-the-box is to
perform the installation via _Anaconda following the below instructions.
To do this, we propose a configuration file (not minimal) that can be 
used as is to set up an environment.

Install Anaconda::

   wget https://repo.anaconda.com/archive/Anaconda3-2018.12-MacOSX-x86_64.sh #(python3.)
   bash Anaconda-latest-Linux-x86_64.sh

Install all dependencies along with _Python 3+.
Once Conda has been installed, the provided (non minimal) conda configuration 
can be used (from _Instructions).
To use the spec file to create an identical environment on the same machine or another machine::

    conda create --name envname --file spec-file.txt

To use the spec file to install its listed packages into an existing environment::

    conda install --name myenv --file spec-file.txt

where :envvar:`spec-file.txt` points toward the provided file originally 
in :envvar:`sknet/config/conda_spec-file.txt`.


.. _Instruction: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#cloning-an-environment
.. _Python: https://www.python.org/download/releases/3.0/
.. _Tensorflow: https://www.tensorflow.org/
.. _Anaconda: https://www.anaconda.com/
