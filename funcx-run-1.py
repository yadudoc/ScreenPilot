# from candle_apps.candle_apps import run_intranode
# from candle_apps.candle_apps import ModelInferer
from funcx.sdk.client import FuncXClient
import time
import argparse
import sys
import os
import logging


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


def warm_endpoint(sleep_time=10):
    """A funcX function to warm the endpoint
    and create workers"""
    import time
    time.sleep(sleep_time)
    return 1


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


def create_output_dir(out_file_path):
    import os
    os.makedirs(out_file_path, exist_ok=True)
    return 'Done' 


def check_done(out_file_path):
    import os
    done_batch = set()
    for f in os.listdir(out_file_path):
        index_start = int(f.split('.')[-2].split('-')[1])
        done_batch.add(index_start)
    return done_batch


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
    res = fxc.run(filename=os.path.join(exec_config['filename'], out_file_name),
                  index=idx,
                  batchsize=batchsize,
                  index_start=0,
                  index_end=1000,
                  workdir=exec_config['workdir'],
                  out_file=f'{exec_config["out_path"]}/{out_file_name.split(".")[0]}_descriptor/{out_file_name}.chunk-{idx_start}-{idx_end}.pkl',
                  endpoint_id=endpoint_uuid,
                  function_id=func_uuid)
    return res


def warm_ep(endpoint_uuid, stats, funcx_warm, warm_limit=10):
    """Check if current workers < max workers.
    If so, launch warming tasks.

    Note: this assumes one worker per block.
    """
    cur_nodes = stats[endpoint_uuid]['managers']
    max_nodes = stats[endpoint_uuid]['max_blocks']
    outstanding_tasks = stats[endpoint_uuid]['outstanding_tasks'].get('RAW', 0)

    warming_jobs = outstanding_tasks - cur_nodes
    # Deal with the case of no outstanding tasks
    if outstanding_tasks == 0:
        warming_jobs = 0
    to_warm = max_nodes - (cur_nodes + warming_jobs)
    to_warm = min(to_warm, warm_limit)

    print(f'max: {max_nodes}, cur: {cur_nodes}, warming: {warming_jobs}, to_warm {to_warm}')
    if to_warm > 0:
        for x in range(0, to_warm):
            print(f"Sending warming function to {endpoint_uuid}")
            res1 = fxc.run(sleep_time=10,
                           endpoint_id=endpoint_uuid,
                           function_id=funcx_warm)


def do_work(batch_index, done_batch, batchsize, func_uuid, funcx_warm, endpoints, out_file_name):
    task_ids = {}
    cur = 0
    while len(batch_index) > 0:
        stats, idle = get_endpoints_status(endpoints)
        for ep, idle_workers in idle.items():
            for _ in range(idle_workers):
                try:
                    idx = batch_index.pop(0)
                    if cur in done_batch:
                        print(f'Batch {cur} is done before, skipping')
                        cur += batchsize
                        continue
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
            # now try warming the rest of the nodes
            warm_ep(ep, stats, funcx_warm, warm_limit=min(10, len(batch_index)))
        time.sleep(70)
    return task_ids


def poll_result(task_id, name='funcx function'):
    while True:
        try:
            res = fxc.get_result(task_id)
            break
        except Exception as e:
            if 'pending' in str(e):
                print(f"Waiting for {name}")
            else:
                print(str(e))
                sys.exit(1)
            time.sleep(5)
            pass
    return res


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
                                                        'out_path': '/home/rchard/src/covid19/ScreenPilot/'},
               'a59434ad-9a25-4378-9682-b5110e4eaa48': {'filename': '/home/rchard/src/covid19/ScreenPilot/ena+db.can',
                                                        'workdir': '/tmp/rchard/',
                                                        'out_path': '/home/rchard/src/covid19/ScreenPilot/'},
               '62186e94-ed69-423a-92c2-4e743383247e': {'filename': '/oasis/projects/nsf/anl117/zhuozhao/data/',
                                                        'workdir': '/tmp/zhuozhao/',
                                                        'out_path': '/oasis/projects/nsf/anl117/zhuozhao/data/'},
               'f061bf6e-8025-475f-b102-0a3fc242164d': {'filename': '/projects/CVD_Research/zhuozhao/data/',
                                                        'workdir': '/tmp/zzli/',
                                                        'out_path': '/projects/CVD_Research/zhuozhao/data/'},
              }

    # Get endpoints
    endpoints = args.endpoints.split(",")
    endpoints = ['709118de-1103-463f-8425-281eb93b55ff',  # Theta covid19
                 '67e95158-8bda-4b1f-a0ef-31a1626eba00',  # Ryan ep
                 'a59434ad-9a25-4378-9682-b5110e4eaa48',  # Ryan ep Theta queue (5 nodes)
                 ]
    endpoints = ['62186e94-ed69-423a-92c2-4e743383247e',  # Comet
                 'f061bf6e-8025-475f-b102-0a3fc242164d',  # Theta CVD_Research
                ]
    print(f"Submitting tasks to endpoints {endpoints}")

    # Funcx client
    fxc = FuncXClient()
    fxc.throttling_enabled = False
    print(fxc.base_url, fxc.throttling_enabled)

    # Batch generator
    batchsize = int(args.batch_size)
    print(f"[Main] Chunksize : {batchsize}")
    batch_generator_uuid = fxc.register_function(generate_batch, description="A funcx function for batch generator")
    res = fxc.run(filename=os.path.join(configs[endpoints[0]]['filename'], args.smile_file),
                  batchsize=batchsize,
                  max_batches=int(args.num_batches),
                  endpoint_id=endpoints[0],
                  function_id=batch_generator_uuid)

    batch_generator = poll_result(res, name='batch_generator')
    # batch_generator = generate_batch(args.smile_file, start=0,
    #                                batchsize=int(args.batch_size),
    #                                max_batches=int(args.num_batches))
    # print(f"The batch generator is {batch_generator}")
    print(f"Got the batch generator, in total {len(batch_generator)} batches")

    # Create output directory and check done batch
    create_out_uuid = fxc.register_function(create_output_dir, description="A funcx function for creating output directory")
    check_done_uuid = fxc.register_function(check_done, description="A funcx function for checking done batches")
    tasks = []
    for ep in endpoints:
        out_file_path = os.path.join(configs[ep]["out_path"], args.smile_file.split(".")[0] + '_descriptor')
        task = fxc.run(out_file_path,
                       endpoint_id=ep,
                       function_id=create_out_uuid)
        tasks.append(task)
    for task in tasks:
        res = poll_result(task, name='creating output directory')
    for ep in endpoints:
        out_file_path = os.path.join(configs[ep]["out_path"], args.smile_file.split(".")[0] + '_descriptor')
        task = fxc.run(out_file_path,
                       endpoint_id=ep,
                       function_id=check_done_uuid)
        tasks.append(task)
    done_batch = set()
    for task in tasks:
        res = poll_result(task, name="checking done batches")
        done_batch.update(res)
    print(f"The done batches are {done_batch}")

    # Register funcx function
    funcx_runner_uuid = fxc.register_function(funcx_runner, description="A funcx function for covid19")

    funcx_warm_uuid = fxc.register_function(warm_endpoint,
                                            description="A funcx warming function for covid19")
    
    # Submit tasks
    task_ids = do_work(batch_generator, done_batch, batchsize, funcx_runner_uuid, funcx_warm_uuid, endpoints, args.smile_file)

    ids = list(task_ids.keys())
    pending_tasks = set(ids)
    failed_results = []
    while len(pending_tasks) > 0:
         print("[Main] Waiting for {} pending tasks...".format(len(pending_tasks)))
         results = fxc.get_batch_status(ids)
         for task_id, res in results.items():
            if task_id in pending_tasks:
                i = task_ids[task_id]
                if res['pending'] == 'False':
                    pending_tasks.remove(task_id)
                    if 'exception' in res:
                        failed_results.append(res)
                        print(f"ChUnk {i} failed")
                    elif 'result' in res:
                        print(f"Chunk {i} is done")
         time.sleep(5)

    print("All done!")
    for res in failed_results:
        print("Raising exception for failed results")
        res['exception'].reraise()
