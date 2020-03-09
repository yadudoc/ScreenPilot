from parsl.config import Config
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_query


config = Config(
    executors=[
        HighThroughputExecutor(
            label='Comet_HTEX_multinode',
            address=address_by_query(),
            max_workers=1,
            provider=SlurmProvider(
                # 'debug',
                'compute',
                launcher=SrunLauncher(),
                # string to prepend to #SBATCH blocks in the submit
                # script to the scheduler
                scheduler_options='source ~/anaconda3/bin/activate; conda activate candle_py3.7',
                # Command to be run before starting a worker, such as:
                # 'module load Anaconda; source activate parsl_env'.
                worker_init='',
                walltime='01:00:00',
                init_blocks=1,
                max_blocks=1,
                nodes_per_block=32,
            ),
        )
    ]
)
