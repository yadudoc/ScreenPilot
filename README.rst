Installation
------------


* Set up a conda env ::
    conda create --name candle python=3.7

* Install requirements ::
    conda install -c rdkit -c mordred-descriptor mordred
    conda install scikit-learn pandas


Notebooks :

`initial_timing_study.ipynb` looks at the cost of serializing and dispatching tasks to a remote worker.

`resilient_workflow.ipynb` looks at creating batches of tasks and handling failures.

Notes::
  Train contains some smiles to play aroudn with. The models are fake here
  since loading up the real models would take some time to do.

  This interface is exactly the same however.
