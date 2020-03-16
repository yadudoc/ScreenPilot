# from candle_apps.candle_apps import run_intranode
# from candle_apps.candle_apps import ModelInferer
from funcx.sdk.client import FuncXClient
import time
import argparse


def generate_batch(filename, start=0, batchsize=10, max_batches=10):
    counter = 0
    if max_batches == 0:
        max_batches = 999999999

    x = 'Hello'
    batch_index = []
    with open(filename) as current:
        batch_index.append(current.tell())
        counter += 1

        while x and counter < max_batches:
            counter += 1
            for i in range(batchsize):
                x = current.readline()

            batch_index.append(current.tell())
        return batch_index


def funcx_runner(index,
                 filename,
                 batchsize, 
                 index_start, 
                 index_end,
                 workdir=None, 
                 out_file=None):

    """funcX function to perform work"""
    from candle_apps.candle_node_local import funcx_node_local
    return funcx_node_local(filename, index, batchsize, index_start, index_end,
                            workdir=workdir, out_file=out_file)


def get_endpoints_status(endpoints):
    endpoint_status = {}
    idle_workers = {}
    for ep in endpoints:
        endpoint_status[ep] = fxc.get_endpoint_status(ep)[0]
        idle_workers[ep] = endpoint_status[ep]['idle_workers']
    return endpoint_status, idle_workers


def submit_job(endpoint_uuid,
               func_uuid,
               batchsize,
               idx,
               idx_start,
               idx_end,
               out_file_name
               ):

    exec_config = configs[endpoint_uuid]
    res = fxc.run(filename=exec_config['filename'],
                  index=idx,
                  batchsize=batchsize,
                  index_start=0,
                  index_end=1000,
                  workdir=exec_config['workdir'],
                  out_file=f'{exec_config["out_path"]}/{out_file_name}.chunk-{idx_start}-{idx_end}.pkl',
                  endpoint_id=endpoint_uuid,
                  function_id=func_uuid)
    return res


def do_work(batch_index, batchsize, func_uuid, endpoints, out_file_name):
    task_ids = {}
    cur = 0
    while len(batch_index) > 0:
        stats, idle = get_endpoints_status(endpoints)
        for ep, idle_workers in idle.items():
            for x in range(idle_workers):
                try:
                    idx = batch_index.pop(0)
                    print(f'submitting {idx} to {ep}')
                    task_id = submit_job(ep,
                                         func_uuid,
                                         batchsize,
                                         idx,
                                         cur,
                                         cur+batchsize,
                                         out_file_name
                                         )
                    task_ids[task_id] = idx
                    cur += batchsize
                except IndexError as e:
                    print('Finished distributing tasks to endpoints!')
                    return task_ids
        time.sleep(1)
    return task_ids


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version",
                        help="Print Endpoint version information")
    parser.add_argument("-d", "--debug", action='store_true',
                        help="Enables debug logging")
    parser.add_argument("-n", "--num_batches", default=8,
                        help="Number of batches to load and run. Default=8, if set to 0 the entire file will be used")
    parser.add_argument("-s", "--smile_file", default="train.csv",
                        help="File path to the smiles csv file")
    parser.add_argument("-b", "--batch_size", default="4",
                        help="Size of the batch of smiles to send to each node for processing. Default=4, should be 10K")
    parser.add_argument("-o", "--outdir", default="outputs",
                        help="Output directory. Default : outputs")
    parser.add_argument("-e", "--endpoints", default="",
                        help="A list of endpoints to run on. Default: local")
    args = parser.parse_args()


    configs = {'709118de-1103-463f-8425-281eb93b55ff': {'filename': '/projects/CVD_Research/zhuozhao/candle/ScreenPilot/ena+db.can',
                                                        'workdir': '/tmp/zzli/',
                                                        'out_path': '/tmp/'},
               '67e95158-8bda-4b1f-a0ef-31a1626eba00': {'filename': '/home/rchard/src/covid19/ScreenPilot/ena+db.can',
                                                        'workdir': '/tmp/rchard/',
                                                        'out_path': '/home/rchard/src/covid19/ScreenPilot/'}
              }

    # Batch generator
    batchsize = int(args.batch_size)
    print(f"[Main] Chunksize : {batchsize}")
    batch_generator = generate_batch(args.smile_file, start=0,
                                     batchsize=int(args.batch_size),
                                     max_batches=int(args.num_batches))
    print(batch_generator)

    # Get endpoints
    endpoints = args.endpoints.split(",")
    endpoints = ['709118de-1103-463f-8425-281eb93b55ff',   #Theta covid19
                 '67e95158-8bda-4b1f-a0ef-31a1626eba00',   #Ryan ep
                ]
    print(f"Submitting tasks to endpoints {endpoints}")

    # Register funcx function
    fxc = FuncXClient()
    fxc.throttling_enabled = False
    print(fxc.base_url, fxc.throttling_enabled)
    funcx_runner_uuid = fxc.register_function(funcx_runner, description="A funcx function for covid19")

    
    # Submit tasks
    task_ids = do_work(batch_generator, batchsize, funcx_runner_uuid, endpoints, args.smile_file)

    ids = list(task_ids.keys())
    pending_tasks = set(ids)
    while len(pending_tasks) > 0:
         print("[Main] Waiting for {} pending tasks...".format(len(pending_tasks)))
         results = fxc.get_batch_status(ids)

         for task_id, res in results.items():
            if task_id in pending_tasks:
                i = task_ids[task_id]
                if res['pending'] == 'False':
                    pending_tasks.remove(task_id)
                    if 'exception' in res:
                        print(str(res['exception']))
                        print(f"ChUnk {i} failed")
                    elif 'result' in res:
                        print(f"Chunk {i} is done")
         time.sleep(5)

    print("All done!")
