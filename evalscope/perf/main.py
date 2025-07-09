import asyncio
import copy
import os
import platform
import threading
import time
from argparse import Namespace
# import uuid # Removed as pipeline_run_id is no longer needed

from evalscope.perf.utils.local_server import start_app
from evalscope.perf.utils.log_utils import init_swanlab, init_wandb
from evalscope.perf.utils.prometheus_util import init_prometheus, save_to_prometheus
from evalscope.utils.logger import configure_logging, get_logger
from evalscope.utils.model_utils import seed_everything
from .arguments import Arguments, parse_args
from .benchmark import benchmark
from .utils.db_util import get_output_path, save_to_sql
from .utils.handler import add_signal_handlers
from .utils.rich_display import print_summary
from .utils.benchmark_util import Metrics # Import Metrics to use its constants

logger = get_logger()


def run_one_benchmark(args: Arguments, output_path: str = None):
    if isinstance(args.parallel, list):
        args.parallel = args.parallel[0]
    if isinstance(args.number, list):
        args.number = args.number[0]

    # Setup logger and output
    args.outputs_dir = output_path

    logger.info('Starting benchmark with args: ')
    logger.info(args)

    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    if platform.system() != 'Windows':
        add_signal_handlers(loop)

    start_timestamp = time.time()
    metrics_result, percentile_result = loop.run_until_complete(benchmark(args))
    end_timestamp = time.time()

    # Push metrics to Prometheus if enabled
    if args.enable_prometheus_metrics:
        save_to_prometheus(args, metrics_result, percentile_result, start_timestamp, end_timestamp)

    # Save metrics to database if enabled
    if args.enable_database_push:
        save_to_sql(args, metrics_result, percentile_result, start_timestamp, end_timestamp)

    return metrics_result, percentile_result


def run_multi_benchmark(args: Arguments, output_path: str = None):
    metric_results = []
    percentile_results = []
    number_list = copy.deepcopy(args.number)
    parallel_list = copy.deepcopy(args.parallel)
    for i, (number, parallel) in enumerate(zip(number_list, parallel_list)):
        args.number = number
        args.parallel = parallel
        # Set up output path for each run
        cur_output_path = os.path.join(output_path, f'parallel_{parallel}_number_{number}')
        os.makedirs(cur_output_path, exist_ok=True)
        # Start the benchmark
        metrics_result, percentile_result = run_one_benchmark(args, output_path=cur_output_path)
        # Save the results
        metric_results.append((metrics_result, percentile_result))
        # Sleep between runs to avoid overwhelming the server
        if i < len(number_list) - 1:
            logger.info('Sleeping for 5 seconds before the next run...')
            time.sleep(5)
    # Analyze results
    print_summary(metric_results, args.model_id)
    return metric_results, percentile_result


def run_perf_benchmark(args):
    # Check if args is a dictionary or Namespace
    if isinstance(args, dict):
        args = Arguments(**args)
    elif isinstance(args, Namespace):
        args = Arguments.from_args(args)

    if args.seed is not None:
        seed_everything(args.seed)

    # Initialize output directory
    output_path = get_output_path(args)
    configure_logging(args.debug, os.path.join(output_path, 'benchmark.log'))

    # Initialize wandb and swanlab
    if args.wandb_api_key:
        init_wandb(args)
    if args.swanlab_api_key:
        init_swanlab(args)

    # Initialize Prometheus metrics here, once per run_perf_benchmark call
    if args.enable_prometheus_metrics:
        init_prometheus(args)

    # Initialize local server if needed
    if args.api.startswith('local'):
        #  start local server
        server = threading.Thread(target=start_app, args=(copy.deepcopy(args), ), daemon=True)
        server.start()
    # Start benchmark
    if len(args.number) == 1:
        return run_one_benchmark(args, output_path=output_path)
    else:
        return run_multi_benchmark(args, output_path=output_path)


if __name__ == '__main__':
    args = Arguments.from_args(parse_args())
    metrics_result, percentile_result = run_perf_benchmark(args)
    print(metrics_result)
    print(percentile_result)
