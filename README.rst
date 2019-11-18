Installation
------------


* Set up a conda env ::
    conda create --name candle python=3.7

* Install requirements ::
    conda install -c rdkit -c mordred-descriptor mordred
    conda install scikit-learn pandas keras

Installing parsl on Theta
-------------------------

On most systems install parsl via pip works::
    pip install parsl

However on some machines, like Theta, the pip module for `psutil` simply won't install properly.
In this situation, it's easier to install the `psutil` module from conda and then do the parsl install::
    conda install psutil
    pip install parsl



Notebooks :

`initial_timing_study.ipynb` looks at the cost of serializing and dispatching tasks to a remote worker.

`resilient_workflow.ipynb` looks at creating batches of tasks and handling failures.

Notes::
  Train contains some smiles to play aroudn with. The models are fake here
  since loading up the real models would take some time to do.

  This interface is exactly the same however.
