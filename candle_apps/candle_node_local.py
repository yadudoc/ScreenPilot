from rdkit import Chem
from mordred import Calculator, descriptors
import pickle
from multiprocessing import Pool, TimeoutError
import time
import os
import shutil
import sys
import logging
import pybel
from mordred import Calculator, descriptors
from rdkit import Chem
import numpy as np
import pickle


calc = Calculator(descriptors, ignore_3D=False) # this object doesn't need to be created everytime. Can make global I think?

def compute_descript(smile, walltime=1):
    # calc = Calculator(descriptors, ignore_3D=True) # this object doesn't need to be created everytime. Can make global I think?
    # ignore_3D set to false, - Xuefeng

    #read smiles
    mol = Chem.MolFromSmiles(smile)

    if mol is None or len(mol.GetAtoms()) > 100:
        print("Error processing mol")
        return pickle.dumps(None)

    descs = calc(mol)

    # Mods from Xuefeng
    descs.fill_missing("nan")

    data = np.array(descs).flatten().astype(np.float32) #could run in FP16 UNO , something to think about
    # data = descs
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



def run_local(smiles, index_start, index_end, out_file=None, logger=None):

    descripts = {}

    # return len(smiles)

    if logger is None:
        if out_file:
            parent = os.path.dirname(out_file)
            try:
                os.makedirs(parent)
            except:
                pass
        else:
            parent = "."

        logger= set_file_logger(f'{parent}.{index_start}-{index_end}.log')
        logger.info("Running with python: {}".format(sys.version))

    try:
        start = time.time()
        logger.info("Starting")
        with Pool(os.cpu_count()) as p:
            logger.info("Created pool with {0} processes for {0} cores".format(os.cpu_count()))
            launched = p.map(compute_descript, smiles)
            for index, s in enumerate(smiles):
                cleaned_s, *drug_id = s.strip().split()
                descripts[cleaned_s] = (drug_id, launched[index])

            #for s, descript in zip(smiles,  p.map(compute_descript, smiles)):
            #   logger.debug("Got descript for {}".format(s))
            #   descripts[s] = descript
    except Exception as e:
        logger.exception("Caught error during compute")
        raise

    delta = time.time() - start
    logger.info("Completed {} in {}s. Throughput: {} Smiles/s".format(len(smiles), delta, len(smiles)/delta))

    with open(out_file, 'wb') as f:
        pickle.dump(descripts, f)

    logger.info(f"Wrote output to {out_file}")
    logger.handlers.pop()
    return out_file


def funcx_node_local(filename, index, batchsize, index_start, index_end,
                     out_file=None, workdir=None):

    if workdir is None:
        workdir = os.environ['WORKDIR']
    os.makedirs(workdir, exist_ok=True)


    logger= set_file_logger(f'{workdir}/test.log')
    logger.info("------------------------------")
    logger.info("Running with python: {}".format(sys.version))
    logger.info("Workdir: {}".format(workdir))

    basename = os.path.basename(filename)
    real_local_path = os.path.expandvars(f'/tmp/$USER/{basename}')

    os.makedirs(os.path.expandvars(f'/tmp/$USER'), exist_ok=True)
    logger.info(f"Real_local_path : {real_local_path}")

    if os.path.exists(real_local_path):
        logger.info(f"Source data file exists at {real_local_path}")
        pass
    elif os.path.exists(filename):
        logger.info(f"Source file is at {filename}, copying to {real_local_path}")
        shutil.copyfile(filename, real_local_path)
    elif os.path.exists(f"{workdir}/{filename}"):
        logger.info(f"Source file is at {workdir}/{filename}, copying to {real_local_path}")
        shutil.copyfile(f"{workdir}/{filename}", real_local_path)
    else:
        logger.error(f"File {filename} was not found as-is, or in workdir:{workdir}/{filename} or {real_local_path}")
        raise Exception(f"File {filename} was not found as-is, or in workdir:{workdir}/{filename} or {real_local_path}")


    with open(real_local_path) as current:
        logger.info(f"Begin Reading {batchsize} items from index:{index}")
        current.seek(index)
        smiles = [current.readline() for i in range(batchsize)]
        logger.info("Smiles loaded")

    out_file = run_local(smiles,
                         index_start,
                         index_end,
                         out_file=out_file,
                         logger=logger)

    return out_file


if __name__ == "__main__":

    import pandas as pd
    print("[Main] Loading all data available")
    # smiles = pd.read_csv("train.csv", nrows=1000).iloc[:,0].tolist()
    # smiles = pd.read_csv("drugbank-all.smi", nrows=10000, error_bad_lines=False)
    count = 3000
    with open("ena+db.can") as f:
        smiles = f.readlines()[:count]
    run_local(smiles, 0, count, f"outputs/outputs-0-{count}.pkl")
