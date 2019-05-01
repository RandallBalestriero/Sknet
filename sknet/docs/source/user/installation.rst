Installation
============

.. rubric:: Requirements
    - _Python 3. 
    - _CuDNN
    - _Tensorflow set up with GPU
    - (h5py, numpy, ...)

.. rubric:: Set-Up Environment variables

Set up the environment variable DATASET_PATH to where you want sknet
to put (when downloaded) all the dataset by default. It should be a valid path.
It can be added in yoyr :file:`.bashrc` as 
``export DATASET_PATH='/home/user/DATASET/'``. This can be done using the 
following command run in a terminal

    echo "export DATASET_PATH='/home/user/DATASET/'" >> ~/.bashrc

where the path can be set to any existing directory.

.. rubric:: Clone Sknet

Run the following command in a terminal::

    git clone https://github.com/RandallBalestriero/Sknet.git


.. rubric:: CUDA Driver installation

One solution to install the lastest driver is ::

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    sudo apt upgrade

then display the available drivers with ``ubuntu-drivers list`` and simply 
install the desired one via ::

    sudo apt install nvidia-driver-VERSION_NUMBER

which will be loaded at the next reboot of the machine. For the last version 
of Tensorflow (>=1.10), it is required to have the lastest driver (>=400).

.. rubric:: Installation (Anaconda)

An easy way to get everything installed and set-up out-of-the-box is to
perform the installation via _Anaconda following the below instructions.
To do this, we propose a configuration file (not minimal) that can be 
used as is to set up an environment.
Download and execute the last version of anaconda as follows::

   wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh #(python3.)
   bash Anaconda-latest-Linux-x86_64.sh


where the last version of anaconda can be obtained online. Now install all the
needed dependencies along with _Python 3+. We provided a non minimal conda 
configuration file that can be used as follow::

    conda create --name envname --file spec-file.txt

to create an environment with all dependencies install. To simply add the 
dependencies to an already existing environment, perform::

    conda install --name myenv --file spec-file.txt

where :envvar:`spec-file.txt` points toward the provided file originally 
in :envvar:`Sknet/sknet/config/conda_spec-file.txt`.


.. _Instruction: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#cloning-an-environment
.. _Python: https://www.python.org/download/releases/3.0/
.. _Tensorflow: https://www.tensorflow.org/
.. _Anaconda: https://www.anaconda.com/
