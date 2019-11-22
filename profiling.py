import pandas as pd
import keras
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import time

'''
Input is a smile. May fail, thats ok.
See MOrdred Documentation https://github.com/mordred-descriptor/mordred
'''
def compute_descript(smile):
    start = time.time()
    calc = Calculator(descriptors, ignore_3D=True) # this object doesn't need to be created everytime. Can make global I think?

    #read smiles
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("Error processing mol")
        return None

    descs = calc(mol)

    x = np.array(descs).flatten().astype(np.float32) #could run in FP16 UNO , something to think about
    print("Delta compute_descript : {}".format(time.time() - start))
    return x


# This will change, but the interface will not.
def combine_drug_features_with_cell_features(vec):
    start = time.time()
    from sklearn.preprocessing import Imputer
    vec_prime = np.zeros((60, vec.shape[0]))
    vec_prime[0] = vec

    #will need to impute missing values
    imp = Imputer()
    vec_prime = imp.fit_transform(vec_prime)
    print("Delta combine_drug... : {}".format(time.time() - start))
    return vec_prime


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

    def __call__(self, batch):
        start = time.time()
        x = self.model.predict(batch, batch_size=batch.shape[0])
        print("Delta infer... : {}".format(time.time() - start))
        return x


if __name__ == '__main__':
    print("Setting up some fake models.")
    models_to_test = [ModelInferer(1613), ModelInferer(1613)]

    print("Loading 2 data")
    smiles = pd.read_csv("train.csv", nrows=100).iloc[:,0].tolist()


    results_dict = {}
    for smile in smiles:
        vec = compute_descript(smile)

        if vec is None:
            continue

        training_batch = combine_drug_features_with_cell_features(vec)

        results = []
        for model in models_to_test:
            results.append(model(training_batch).flatten())
        results_dict[smile] = results

    print("Done. ")
    print(results_dict)
