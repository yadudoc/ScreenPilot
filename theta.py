from parsl.config import Config
from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname
#from parsl.monitoring.monitoring import MonitoringHub

launch_cmd = ("bash /projects/candle_aesp/yadu/unpack_and_load.sh tmp candle_py3.7 /projects/candle_aesp/yadu/candle_py3.7.tar.gz /dev/shm/ ;"
                   "source /dev/shm/candle_py3.7/bin/activate ;"
                   "cd /projects/candle_aesp/yadu/ScreenPilot/; python3 setup.py install; "
                   "which python3; \n"
                   "process_worker_pool.py {debug} {max_workers} "
                   "-p {prefetch_capacity} "
                   "-c {cores_per_worker} "
                   "-m {mem_per_worker} "
                   "--poll {poll_period} "
                   "--task_url={task_url} "
                   "--result_url={result_url} "
                   "--logdir={logdir} "
                   "--block_id={{block_id}} "
                   "--hb_period={heartbeat_period} "
                   "--hb_threshold={heartbeat_threshold} ")


config = Config(
    executors=[
        HighThroughputExecutor(
            label='theta_local_htex_multinode',
            max_workers=1, # The target process itself if a Multiprocessing application. We do not
            # need to overload the compute node with parsl workers.
            address="10.236.1.195",
            # address=address_by_hostname(),
            # launch_cmd=launch_cmd,
            prefetch_capacity=2,
            provider=CobaltProvider(
                #queue='debug-flat-quad',
                #queue='default',
                queue='CVD_Research',
                #account='candle_aesp',
                account='CVD_Research',
                launcher=AprunLauncher(overrides=" -d 64"),
                walltime='24:00:00',
                nodes_per_block=150,
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
                # string to prepend to #COBALT blocks in the submit
                # script to the scheduler eg: '#COBALT -t 50'
                # scheduler_options='',
                # Command to be run before starting a worker, such as:
                # 'module load Anaconda; source activate parsl_env'.
                # worker_init='source ~/Anaconda/bin/activate; conda activate candle_py3.7;',
                worker_init='source ~/anaconda3/bin/activate; conda activate candle_py3.7;',
                #worker_init=("bash /projects/candle_aesp/yadu/unpack_and_load.sh tmp candle_py3.7 /projects/candle_aesp/yadu/candle_py3.7.tar.gz /dev/shm/ ;"
                #             "source /dev/shm/candle_py3.7/bin/activate ;"
                #             "cd /projects/candle_aesp/yadu/ScreenPilot/; python3 setup.py install; "
                #             "which python3; \n"),
                cmd_timeout=300,
            ),
        )
    ],
    strategy=None,
)


"""
                worker_init='''
ENV_NAME="candle_env"
CONDA_FILE="~/candle_conda.tar.gz"
DEST="/dev/shm/yadu"

echo "Loading env:$ENV_NAME from $CONDA_FILE unpacked at $DEST"

if [ -d $DEST/$ENV_NAME ]
then
    echo "Env already exists at $DEST/$ENV_NAME... Reusing"
else
    echo "Copying and untarring at $DEST/$ENV_NAME"
    mkdir -p $DEST/$ENV_NAME
    tar -xzf $CONDA_FILE -C $DEST/$ENV_NAME
fi

source $DEST/$ENV_NAME/bin/activate

#source ~/unpack_and_load.sh candle_env ~/candle_conda.tar.gz /dev/shm/yadu''',

"""
