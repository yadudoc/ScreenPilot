from rdkit import Chem
from mordred import Calculator, descriptors
import pickle
from multiprocessing import Pool, TimeoutError
import time
import os
import logging

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
    # return pickle.dumps(data) # We do this to avoid a bug in the serialization routines that Parsl
    return data


def set_file_logger(filename: str, name: str = 'candle', level: int = logging.DEBUG, format_string = None):
    """Add a stream log handler.

    Args:
        - filename (string): Name of the file to write logs to
        - name (string): Logger name
        - level (logging.LEVEL): Set the logging level.
        - format_string (string): Set the format string

    Returns:
       -  None
    """
    if format_string is None:
        format_string = "%(asctime)s.%(msecs)03d %(name)s:%(lineno)d [%(levelname)s]  %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(level)
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger



def run_local(smiles, index_start, index_end, out_file=None):

    descripts = {}

    if out_file:
        parent = os.path.dirname(out_file)
        try:
            os.makedirs(parent)
        except:
            pass
    else:
        parent = "."
    logger= set_file_logger(f'{parent}.{index_start}-{index_end}.log')

    start = time.time()
    logger.info("Starting")
    with Pool(os.cpu_count()) as p:
        for s, descript in zip(smiles,  p.map(compute_descript, smiles)):
            descripts[s] = descript

    with open(out_file, 'wb') as f:
        pickle.dump(descripts, f)

    delta = time.time() - start
    logger.info("Completed {} in {}s. Throughput: {} Smiles/s".format(len(smiles), delta, len(smiles)/delta))
    logger.info(f"Wrote output to {out_file}")
    return descripts


if __name__ == "__main__":

    import pandas as pd
    print("[Main] Loading all data available")
    smiles = pd.read_csv("train.csv", nrows=100).iloc[:,0].tolist()
    run_local(smiles, 0, 100, "outputs/outputs-0-100.pkl")
