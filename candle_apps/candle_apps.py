import pandas as pd
import keras
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import Imputer
import pickle

from parsl import python_app
@python_app
def compute_descript(smile, walltime=1):
    """
    import random
    import time
    if random.randint(0,8) == 0:
        time.sleep(1)
    """
    from mordred import Calculator, descriptors
    from rdkit import Chem
    import numpy as np
    import pickle

    calc = Calculator(descriptors, ignore_3D=True) # this object doesn't need to be created everytime. Can make global I think?

    #read smiles
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("Error processing mol")
        return pickle.dumps(None)

    descs = calc(mol)

    data = np.array(descs).flatten().astype(np.float32) #could run in FP16 UNO , something to think about
    return pickle.dumps(data) # We do this to avoid a bug in the serialization routines that Parsl


# This will change, but the interface will not.
@python_app
def combine_drug_features_with_cell_features(pkl_vec, walltime=1):
    import pickle
    import numpy as np
    from sklearn.preprocessing import Imputer

    vec = pickle.loads(pkl_vec)
    if vec is None:
        return pickle.dumps(None)

    vec_prime = np.zeros((60, vec.shape[0]))
    vec_prime[0] = vec

    #will need to impute missing values
    imp = Imputer()
    vec_prime = imp.fit_transform(vec_prime)

    pkl_data = pickle.dumps(vec_prime)
    return pkl_data


#The model will change, the inferface will not.
class ModelInferer():
    def __init__(self, vec_size):
        from keras.layers import Dense
        from keras.models import Model, Input
        input = Input((vec_size,))
        h = Dense(32)(input)
        out = Dense(1)(h)
        self.model = Model(input,out)
        self.model.compile(optimizer='sgd', loss='mse')


@python_app
def infer(pkl_model, pkl_batch):
    import pickle
    batch = pickle.loads(pkl_batch)
    model = pickle.loads(pkl_model)
    if batch is None:
        return None

    inferred = model.predict(batch, batch_size=batch.shape[0])
    return inferred



def launch_tasks(smiles, models_to_test, chunksize=10, wait=True):
    """
     Launch tasks on chunks of data

     Parsl does batching internally, but we can do better!

     We have an estimate of the runtime for a batch of N tasks, and we use that to our advantage by creating
     chunks of "smiles" that are dispatched to the now batched, `app_compute_descript_batches` function.

     `chunksize` is configurable. In a smarter version we could tie `chunksize` and `walltime` together.
    """
    descript_chunks = {}
    combine_chunks = {}
    infer_chunks = {}

    for i, smile in enumerate(smiles):
        descript_vecs_list = compute_descript(smile)
        descript_chunks[i] = descript_vecs_list
        training_batch_list = combine_drug_features_with_cell_features(descript_vecs_list)
        combine_chunks[i] = training_batch_list

        """
        x = infer(pickle.dumps(models_to_test[0]), training_batch_list)
        infer_chunks[smile] = x
        """
        infer_chunks[smile] = {}
        for idx, model in enumerate(models_to_test):
            x = infer(pickle.dumps(model.model), training_batch_list)
            infer_chunks[smile][idx] = x

    #return infer_chunks[smile][0].result()
    print(infer_chunks)
    final_results = {}
    for smile in infer_chunks:
        final_results[smile] = {}
        for model in infer_chunks[smile]:
            try:
                final_results[smile][model] = infer_chunks[smile][model].result()
            except Exception as e:
                final_results[smile][model] = "Failed due to {}".format(e)

    return final_results

@python_app
def run_intranode(smiles, pkl_models):
    import parsl
    from parsl.configs.htex_local import config
    config.executors[0].label = "theta_intranode"
    parsl.load(config)

    models_to_test = pickle.loads(pkl_models)
    result_chunks = launch_tasks(smiles, models_to_test, chunksize=1)
    return result_chunks



if __name__ == '__main__':

    print("Testing single node pipeline")
    print("Setting up some fake models.")

    import parsl
    from parsl.configs.htex_local import config
    parsl.load(config)

    models_to_test = [ModelInferer(1613), ModelInferer(1613)]
    # models_to_test = [1, 2]

    print("Loading 2 data")
    smiles = pd.read_csv("train.csv", nrows=2).iloc[:,0].tolist()

    descript_steps, result_chunks = launch_tasks(smiles, models_to_test, chunksize=1)
    print("Final : ", result_chunks)
