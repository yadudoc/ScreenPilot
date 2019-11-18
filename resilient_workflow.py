#!/usr/bin/env python
import argparse
import pandas as pd
import keras
import numpy as np
import time
import parsl
import pickle
from parsl import python_app
from parsl.app.errors import AppTimeout
from parsl.configs.htex_local import config


# ### Update descript to process batches
#
# We want the descript step to consume batches of smiles to minimize the task launch costs.
# Here we add a `@python_app` decorator that marks this function for remote/distributed execution.
#
# Key point to note is that we add a special `walltime=<int:seconds>` kwarg, that causes the function to raise a `parsl.app.errors.AppTimeout` exception if the function runs beyond the set walltime.

# In[5]:


@python_app
def app_compute_descript_batches(smile_list, walltime=1):
    """ Takes a list of smiles and returns a corresponding list of descs.
    """
    from mordred import Calculator, descriptors
    from rdkit import Chem
    import numpy as np
    import pickle
    # this object doesn't need to be created everytime. Can make global I think?
    calc = Calculator(descriptors, ignore_3D=True)

    results_list = []
    for smile in smile_list:
        #read smiles
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print("Error processing mol")
            result = None
        else:
            descs = calc(mol)
            result = pickle.dumps(np.array(descs).flatten().astype(np.float32))

        results_list.append(result)

    return results_list


# This will change, but the interface will not.
@python_app
def combine_drug_features_with_cell_features(vec_list):
    from sklearn.preprocessing import Imputer
    import numpy as np
    import pickle
    results = []
    for b_vec in vec_list:
        vec = pickle.loads(b_vec)
        vec_prime = np.zeros((60, vec.shape[0]))
        vec_prime[0] = vec

        #will need to impute missing values
        imp = Imputer()
        vec_prime = imp.fit_transform(vec_prime)
        results.append(pickle.dumps(vec_prime)) # <-- Another serialization pain point

    return results



def launch_tasks(data, chunksize=10):
    """
     Launch tasks on chunks of data

     Parsl does batching internally, but we can do better!

     We have an estimate of the runtime for a batch of N tasks, and we use that to our advantage by creating
     chunks of "smiles" that are dispatched to the now batched, `app_compute_descript_batches` function.

     `chunksize` is configurable. In a smarter version we could tie `chunksize` and `walltime` together.
    """
    proc_chunks = {}
    result_chunks = {}
    for i in range(1, len(data), chunksize):
        chunk = data[i:i+chunksize]
        descript_vecs_list = app_compute_descript_batches(chunk)
        training_batch_list = combine_drug_features_with_cell_features(descript_vecs_list)
        proc_chunks[i] = descript_vecs_list
        result_chunks[i] = training_batch_list
    return proc_chunks, result_chunks




def pipeline(smiles):
    # Initial launch of all tasks
    proc_chunks, result_chunks = launch_tasks(smiles)

    chunksize=10
    for key in proc_chunks:
        try:
            x = proc_chunks[key].result()
        except AppTimeout as e:
            print("Caught timeout for chunk index: {}:{}".format(key,key+chunksize))


    print("Proc chunks", proc_chunks)

    unpacked = {}
    unpacked_tail = {}
    for key in proc_chunks:
        try:
            x = proc_chunks[key].result()
        except AppTimeout as e:
            print("Launching unpacked tasks: {}:{}".format(key,key+chunksize))
            unpacked[key], unpacked_tail[key] = launch_tasks(smiles[key:key+chunksize], chunksize=1)

    for key in unpacked:
        print("Peeking inside batch {}:{} ------------".format(key, key+chunksize))
        for item in unpacked[key]:
            print("   Item {}".format(item))
            print(unpacked[key][item])
        print("---------------------------------------")


    print(unpacked_tail)
    for batch in unpacked_tail:
        for item in unpacked_tail[batch]:
            print(unpacked_tail[batch][item])

    x = unpacked_tail[71][1].result()

    r = pickle.loads(x[0])
    print(r)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version",
                        help="Print Endpoint version information")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Enables debug logging")
    args = parser.parse_args()

    # Most of the app that hit the timeout will complete if retried.
    # but for this demo, I'm not setting retries.
    # config.retries = 2
    parsl.load(config)

    if args.debug:
        parsl.set_stream_logger()

    print("Loading all data available")
    smiles = pd.read_csv("train.csv", nrows=158).iloc[:,0].tolist()
    print("Total of {} available".format(len(smiles)))

    pipeline(smiles)
