# from candle_apps.candle_apps import run_intranode
# from candle_apps.candle_apps import ModelInferer
# from candle_apps.candle_node_local import run_local
import pandas as pd
import argparse
import os
import parsl
import pickle
from collections import OrderedDict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version",
                        help="Print Endpoint version information")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Enables debug logging")
    parser.add_argument("-n", "--num_smiles", default=8,
                        help="Number of smiles to load and run. Default=10000, if set to 0 the entire file will be used")
    parser.add_argument("-s", "--smile_file", default="train.csv",
                        help="File path to the smiles csv file")
    parser.add_argument("-b", "--batch_size", default="4",
                        help="Size of the batch of smiles to send to each node for processing. Default=4, should be 10K")
    parser.add_argument("-o", "--outdir", default="outputs",
                        help="Output directory. Default : outputs")
    parser.add_argument("-c", "--config", default="local",
                        help="Parsl config defining the target compute resource to use. Default: local")
    args = parser.parse_args()

    if args.config == "local":
        from parsl.configs.htex_local import config
        from parsl.configs.htex_local import config
        config.executors[0].label = "Foo"
        config.executors[0].max_workers = 4
    elif args.config == "theta":
        from theta2 import config
    elif args.config == "comet":
        from comet import config

    # Most of the app that hit the timeout will complete if retried.
    # but for this demo, I'm not setting retries.
    # config.retries = 2
    parsl.load(config)

    from candle_apps.candle_node_local import compute_descript
    # parsl_runner = parsl.python_app(run_local)
    parsl_runner = parsl.python_app(compute_descript)

    if args.debug:
        parsl.set_stream_logger()


    with open(args.smile_file) as f:
        if int(args.num_smiles) > 0:
            smiles = f.readlines()[:int(args.num_smiles)]
        else:
            smiles = f.readlines()

    try:
        os.makedirs(args.outdir)
    except:
        pass
    #models_to_test = [ModelInferer(1613), ModelInferer(1613)]
    models_to_test = [1, 2]

    chunksize = int(args.batch_size)
    print(f"[Main] Chunksize : {chunksize}")
    batch_futures = OrderedDict()

    smile_drug_map = {}

    count = 0 

    for i in range(0, len(smiles), chunksize):
        #result_chunks = run_intranode(smiles[i:i+chunksize],
        #                              pickle.dumps(models_to_test))
        outfile = f"{args.outdir}/{args.smile_file}.chunk-{i}-{i+chunksize}.pkl"

        if os.path.exists(outfile):
            # print(f"Batch {outfile} exists. Skipping")
            pass
        else:
            print(f"Batch {outfile} missing")
            counter = 0
            batch_futures[i] = {'futures' : [],
                                'outfile' : outfile}

            smile_drug_map[i] = {}

            for smile in smiles[i:i+chunksize]:
                # print(smile)                
                cleaned_s, *drug_id = smile.strip().split()
                fut = parsl_runner(smile, walltime=30, pickle=True)
                batch_futures[i]['futures'].append(fut)
                smile_drug_map[i][fut] = [cleaned_s, drug_id]

                counter += 1
            count += 1


    for batch_id in batch_futures:        
        print(smile_drug_map[batch_id])

        result_dict = {}
        for fut in smile_drug_map[batch_id]:
            smile, drug = smile_drug_map[batch_id][fut]

            try:
                p_desc = fut.result()
                desc = pickle.loads(p_desc)

            except Exception as e:
                print("Caught exception for smile : [{}]".format(smile_drug_map[batch_id][fut][0]))
                desc = None

            finally:
                result_dict[smile] = (drug, desc)

        outfile = batch_futures[batch_id]['outfile']
        print("Writing results into failed/{}".format(outfile))
        with open('failed/' + outfile, 'wb') as f:
            pickle.dump(result_dict, f)


    parsl.dfk().wait_for_current_tasks()
    print(f"Total missing : {count}")
    """
    print("[Main] Waiting for {} futures...".format(len(batch_futures)))
    for i in batch_futures:
        try:
            x = batch_futures[i].result()
            print(f"Chunk {i} is done with {x}")
        except Exception as e:
            print(f"Exception : {e}")
            print(f"Chunk {i} failed")
    """
    print("All done!")

