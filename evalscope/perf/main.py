import asyncio
import copy
import os
import platform
import threading
import time
from argparse import Namespace
# import uuid # Removed as pipeline_run_id is no longer needed

from evalscope.perf.utils.local_server import start_app
from evalscope.perf.utils.log_utils import init_swanlab, init_wandb, init_prometheus
from evalscope.utils.logger import configure_logging, get_logger
from evalscope.utils.model_utils import seed_everything
from .arguments import Arguments, parse_args
from .benchmark import benchmark
from .utils.db_util import get_output_path
from .utils.handler import add_signal_handlers
from .utils.rich_display import print_summary
from .utils.db_util import save_to_sql
from .utils.benchmark_util import Metrics # Import Metrics to use its constants

logger = get_logger()


# run_one_benchmark no longer needs pipeline_run_id
def run_one_benchmark(args: Arguments, output_path: str = None, prometheus_metrics: dict = None):

    def _restructure_percentiles(percentile_result: dict) -> dict:
        if not percentile_result or 'Percentiles' not in percentile_result:
            return {}

        # Extract percentile labels like ['10', '25', ...]
        percentile_labels = [p.replace('%', '') for p in percentile_result.get('Percentiles', [])]
        flat_percentiles = {}

        for key, values in percentile_result.items():
            if key == 'Percentiles':
                continue
            
            # Sanitize the metric name to be a valid column name component
            sanitized_key = key.replace(' (s)', '_seconds').replace(' (tok/s)', '_tok_per_s').replace(' ', '_')
            
            for i, label in enumerate(percentile_labels):
                if i < len(values):
                    flat_key = f"p{label}_{sanitized_key}"
                    flat_percentiles[flat_key] = values[i]

        return flat_percentiles

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

    # Push metrics to Prometheus if enabled and prometheus_metrics are provided
    if args.enable_prometheus_metrics and prometheus_metrics:
        save_to_prometheus(args, metrics_result, percentile_result, prometheus_metrics, start_timestamp, end_timestamp)

    # Save metrics to database if enabled
    if args.enable_database_push:
        save_to_sql(args, metrics_result, percentile_result, start_timestamp, end_timestamp)

    return metrics_result, percentile_result


def save_to_prometheus(args: Arguments, metrics_result: dict, percentile_result: dict, prometheus_metrics: dict, start_timestamp: float, end_timestamp: float):
    base_labels = {
        'model': args.model_id,
        'concurrency': str(args.parallel),
        'dataset': args.dataset,
        'ep': str(args.ep) if args.ep is not None else '',
        'dp': str(args.dp) if args.dp is not None else '',
        'tp': str(args.tp) if args.tp is not None else '',
        'pd': args.pd if args.pd is not None else '',
        'metadata': args.metadata if args.metadata is not None else '',
    }
    _push_metrics_to_prometheus(args, metrics_result, percentile_result, prometheus_metrics, start_timestamp, end_timestamp, base_labels)

# _push_metrics_to_prometheus no longer needs pipeline_run_id
def _push_metrics_to_prometheus(args: Arguments, metrics_result: dict, percentile_result: dict, prometheus_metrics: dict, start_timestamp: float, end_timestamp: float, base_labels: dict):
    if not args.enable_prometheus_metrics or not args.prometheus_pushgateway_url:
        return

    try:
        from prometheus_client import push_to_gateway
    except ImportError:
        logger.error('prometheus_client is not installed. Cannot push metrics to Pushgateway.')
        return

    registry = prometheus_metrics['registry']
    requests_per_second_gauge = prometheus_metrics['requests_per_second']
    latency_avg_seconds_gauge = prometheus_metrics['latency_avg_seconds']
    ttft_avg_seconds_gauge = prometheus_metrics['ttft_avg_seconds']
    output_tokens_per_second_gauge = prometheus_metrics['output_tokens_per_second']
    total_tokens_per_second_gauge = prometheus_metrics['total_tokens_per_second']
    duration_seconds_gauge = prometheus_metrics['duration_seconds']
    input_tokens_avg_gauge = prometheus_metrics['input_tokens_avg']
    output_tokens_avg_gauge = prometheus_metrics['output_tokens_avg']
    error_rate_gauge = prometheus_metrics['error_rate_gauge']
    latency_percentiles_gauge = prometheus_metrics['latency_percentiles']
    requests_failed_total_counter = prometheus_metrics['requests_failed_total']

    # NEW: Get additional metrics
    avg_time_per_output_token_gauge = prometheus_metrics['avg_time_per_output_token']
    avg_package_latency_gauge = prometheus_metrics['avg_package_latency']
    avg_package_per_request_gauge = prometheus_metrics['avg_package_per_request']
    total_requests_counter = prometheus_metrics['total_requests_counter']
    succeed_requests_counter = prometheus_metrics['succeed_requests_counter']


    # Dynamically construct the job name based on deployment parameters
    job_name_parts = [args.prometheus_job_name] # Start with the base job name
    if args.model_id:
        job_name_parts.append(f"model_{args.model_id.replace('/', '_').replace('-', '_').replace('.', '_')}")
    if args.dataset:
        job_name_parts.append(f"dataset_{args.dataset.replace('/', '_').replace('-', '_').replace('.', '_')}")
    if args.ep is not None:
        job_name_parts.append(f"ep_{args.ep}")
    if args.dp is not None:
        job_name_parts.append(f"dp_{args.dp}")
    if args.tp is not None:
        job_name_parts.append(f"tp_{args.tp}")
    if args.pd:
        job_name_parts.append(f"pd_{args.pd.replace('/', '_').replace('-', '_').replace('.', '_')}")
    
    dynamic_job_name = "_".join(job_name_parts)
    dynamic_job_name = "".join(c for c in dynamic_job_name if c.isalnum() or c == '_')


    

    # Set metric values for gauges with common labels
    if Metrics.REQUEST_THROUGHPUT in metrics_result:
        requests_per_second_gauge.labels(**base_labels).set(metrics_result[Metrics.REQUEST_THROUGHPUT])
    if Metrics.AVERAGE_LATENCY in metrics_result:
        latency_avg_seconds_gauge.labels(**base_labels).set(metrics_result[Metrics.AVERAGE_LATENCY])
    if Metrics.AVERAGE_TIME_TO_FIRST_TOKEN in metrics_result:
        ttft_avg_seconds_gauge.labels(**base_labels).set(metrics_result[Metrics.AVERAGE_TIME_TO_FIRST_TOKEN])
    if Metrics.OUTPUT_TOKEN_THROUGHPUT in metrics_result:
        output_tokens_per_second_gauge.labels(**base_labels).set(metrics_result[Metrics.OUTPUT_TOKEN_THROUGHPUT])
    if Metrics.TOTAL_TOKEN_THROUGHPUT in metrics_result:
        total_tokens_per_second_gauge.labels(**base_labels).set(metrics_result[Metrics.TOTAL_TOKEN_THROUGHPUT])
    if Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST in metrics_result:
        input_tokens_avg_gauge.labels(**base_labels).set(metrics_result[Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST])
    if Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST in metrics_result:
        output_tokens_avg_gauge.labels(**base_labels).set(metrics_result[Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST])
    # Error rate is calculated from failed requests, so it's already covered by requests_failed_total_counter
    # if 'error_rate' in metrics_result: # This was a direct key, now use Metrics.FAILED_REQUESTS
    #     error_rate_gauge.labels(**base_labels).set(metrics_result['error_rate'])

    # Handle counters
    if Metrics.FAILED_REQUESTS in metrics_result:
        requests_failed_total_counter.labels(**base_labels).inc(metrics_result[Metrics.FAILED_REQUESTS])
        # Calculate error rate from failed and total requests
        total_req = metrics_result.get(Metrics.TOTAL_REQUESTS, 0)
        failed_req = metrics_result.get(Metrics.FAILED_REQUESTS, 0)
        if total_req > 0:
            error_rate_gauge.labels(**base_labels).set(failed_req / total_req)
        else:
            error_rate_gauge.labels(**base_labels).set(0) # No requests, no error rate

    if Metrics.TOTAL_REQUESTS in metrics_result:
        total_requests_counter.labels(**base_labels).inc(metrics_result[Metrics.TOTAL_REQUESTS])
    if Metrics.SUCCEED_REQUESTS in metrics_result:
        succeed_requests_counter.labels(**base_labels).inc(metrics_result[Metrics.SUCCEED_REQUESTS])


    # Set percentile metrics
    if percentile_result and 'latency' in percentile_result:
        for quantile, value in percentile_result['latency'].items():
            latency_percentiles_gauge.labels(**base_labels, quantile=str(quantile)).set(value)

    # Set duration gauge
    duration = end_timestamp - start_timestamp
    duration_seconds_gauge.labels(**base_labels, start_timestamp=str(int(start_timestamp)), end_timestamp=str(int(end_timestamp))).set(duration)

    # NEW: Set additional metrics
    if Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN in metrics_result:
        avg_time_per_output_token_gauge.labels(**base_labels).set(metrics_result[Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN])
    if Metrics.AVERAGE_PACKAGE_LATENCY in metrics_result:
        avg_package_latency_gauge.labels(**base_labels).set(metrics_result[Metrics.AVERAGE_PACKAGE_LATENCY])
    if Metrics.AVERAGE_PACKAGE_PER_REQUEST in metrics_result:
        avg_package_per_request_gauge.labels(**base_labels).set(metrics_result[Metrics.AVERAGE_PACKAGE_PER_REQUEST])


    try:
        # Use the dynamically constructed job name
        push_to_gateway(args.prometheus_pushgateway_url, job=dynamic_job_name, registry=registry)
        logger.info(f"Successfully pushed metrics to Prometheus Pushgateway: {args.prometheus_pushgateway_url} with job: {dynamic_job_name}")
    except Exception as e:
        logger.error(f"Failed to push metrics to Prometheus Pushgateway with job {dynamic_job_name}: {e}")


# run_multi_benchmark no longer needs pipeline_run_id
def run_multi_benchmark(args: Arguments, output_path: str = None, prometheus_metrics: dict = None):
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
        # Start the benchmark, passing prometheus_metrics
        metrics_result, percentile_result = run_one_benchmark(args, output_path=cur_output_path, prometheus_metrics=prometheus_metrics)
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

    prometheus_metrics = None
    # Initialize Prometheus metrics here, once per run_perf_benchmark call
    if args.enable_prometheus_metrics:
        prometheus_metrics = init_prometheus(args)

    # Initialize local server if needed
    if args.api.startswith('local'):
        #  start local server
        server = threading.Thread(target=start_app, args=(copy.deepcopy(args), ), daemon=True)
        server.start()
    # Start benchmark
    if len(args.number) == 1:
        return run_one_benchmark(args, output_path=output_path, prometheus_metrics=prometheus_metrics)
    else:
        return run_multi_benchmark(args, output_path=output_path, prometheus_metrics=prometheus_metrics)


if __name__ == '__main__':
    args = Arguments.from_args(parse_args())
    metrics_result, percentile_result = run_perf_benchmark(args)
    print(metrics_result)
    print(percentile_result)
