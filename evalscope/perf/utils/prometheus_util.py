import time
from typing import Dict

from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.utils.logger import get_logger

logger = get_logger()

# Module-level global variable to hold Prometheus metrics
PROMETHEUS_METRICS = None

def init_prometheus(args: Arguments):
    """
    Initializes and returns a dictionary of Prometheus metric objects.
    Each metric is a Gauge, Counter, or other Prometheus metric type.
    """
    
    global PROMETHEUS_METRICS
    base_labels = ['model', 'concurrency', 'dataset', 'ep', 'dp', 'tp', 'pd', 'metadata']
    
    registry = CollectorRegistry()
    # --- Core Performance Metrics (Gauges) ---
    requests_per_second = Gauge(
        'requests_per_second',
        'Request throughput (req/s).',
        base_labels,
        registry=registry
    )
    latency_avg_seconds = Gauge(
        'latency_seconds_avg',
        'Average request latency in seconds.',
        base_labels,
        registry=registry
    )
    ttft_avg_seconds = Gauge(
        'ttft_seconds_avg',
        'Average time to first token in seconds.',
        base_labels,
        registry=registry
    )
    output_tokens_per_second = Gauge(
        'output_tokens_per_second',
        'Output token throughput (tok/s).',
        base_labels,
        registry=registry
    )
    total_tokens_per_second = Gauge(
        'total_tokens_per_second',
        'Total token throughput (tok/s), including prompt tokens.',
        base_labels,
        registry=registry
    )

    # --- Key Statistical Data (Gauges) ---
    duration_seconds = Gauge(
        'duration_seconds',
        'Duration of the benchmark run in seconds.',
        base_labels + ['start_timestamp', 'end_timestamp'],
        registry=registry
    )
    input_tokens_avg = Gauge(
        'input_tokens_per_request_avg',
        'Average input tokens per request.',
        base_labels,
        registry=registry
    )
    output_tokens_avg = Gauge(
        'output_tokens_per_request_avg',
        'Average output tokens per request.',
        base_labels,
        registry=registry
    )
    error_rate_gauge = Gauge(
        'error_rate',
        'Error rate of requests.',
        base_labels,
        registry=registry
    )

    # --- Percentile Metrics (Gauge with quantile label) ---
    latency_percentiles = Gauge(
        'latency_seconds',
        'Request latency percentiles in seconds.',
        base_labels + ['quantile'],
        registry=registry
    )

    # --- Failure Counter ---
    requests_failed_total = Counter(
        'requests_failed_total',
        'Total number of failed requests.',
        base_labels,
        registry=registry
    )

    # --- NEW: Additional Metrics from benchmark_util.py (Gauges) ---
    avg_time_per_output_token = Gauge(
        'avg_time_per_output_token_seconds',
        'Average time per output token in seconds.',
        base_labels,
        registry=registry
    )
    avg_package_latency = Gauge(
        'avg_package_latency_seconds',
        'Average package latency in seconds.',
        base_labels,
        registry=registry
    )
    avg_package_per_request = Gauge(
        'avg_package_per_request',
        'Average number of packages per request.',
        base_labels,
        registry=registry
    )

    # --- NEW: Total Request Counters ---
    total_requests_counter = Counter(
        'total_requests_total',
        'Total number of requests.',
        base_labels,
        registry=registry
    )
    succeed_requests_counter = Counter(
        'succeed_requests_total',
        'Total number of succeed requests.',
        base_labels,
        registry=registry
    )

    PROMETHEUS_METRICS = {
        'registry': registry,
        'requests_per_second': requests_per_second,
        'latency_avg_seconds': latency_avg_seconds,
        'ttft_avg_seconds': ttft_avg_seconds,
        'output_tokens_per_second': output_tokens_per_second,
        'total_tokens_per_second': total_tokens_per_second,
        'duration_seconds': duration_seconds,
        'input_tokens_avg': input_tokens_avg,
        'output_tokens_avg': output_tokens_avg,
        'error_rate_gauge': error_rate_gauge,
        'latency_percentiles': latency_percentiles,
        'requests_failed_total': requests_failed_total,
        'avg_time_per_output_token': avg_time_per_output_token,
        'avg_package_latency': avg_package_latency,
        'avg_package_per_request': avg_package_per_request,
        'total_requests_counter': total_requests_counter,
        'succeed_requests_counter': succeed_requests_counter,
    }

def save_to_prometheus(args: Arguments, metrics_result: dict, percentile_result: dict, start_timestamp: float, end_timestamp: float):
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
    _push_metrics_to_prometheus(args, metrics_result, percentile_result, start_timestamp, end_timestamp, base_labels)

def _push_metrics_to_prometheus(args: Arguments, metrics_result: dict, percentile_result: dict, start_timestamp: float, end_timestamp: float, base_labels: dict):
    if not args.enable_prometheus_metrics or not args.prometheus_pushgateway_url or PROMETHEUS_METRICS is None:
        return

    registry = PROMETHEUS_METRICS['registry']
    requests_per_second_gauge = PROMETHEUS_METRICS['requests_per_second']
    latency_avg_seconds_gauge = PROMETHEUS_METRICS['latency_avg_seconds']
    ttft_avg_seconds_gauge = PROMETHEUS_METRICS['ttft_avg_seconds']
    output_tokens_per_second_gauge = PROMETHEUS_METRICS['output_tokens_per_second']
    total_tokens_per_second_gauge = PROMETHEUS_METRICS['total_tokens_per_second']
    duration_seconds_gauge = PROMETHEUS_METRICS['duration_seconds']
    input_tokens_avg_gauge = PROMETHEUS_METRICS['input_tokens_avg']
    output_tokens_avg_gauge = PROMETHEUS_METRICS['output_tokens_avg']
    error_rate_gauge = PROMETHEUS_METRICS['error_rate_gauge']
    latency_percentiles_gauge = PROMETHEUS_METRICS['latency_percentiles']
    requests_failed_total_counter = PROMETHEUS_METRICS['requests_failed_total']

    # NEW: Get additional metrics
    avg_time_per_output_token_gauge = PROMETHEUS_METRICS['avg_time_per_output_token']
    avg_package_latency_gauge = PROMETHEUS_METRICS['avg_package_latency']
    avg_package_per_request_gauge = PROMETHEUS_METRICS['avg_package_per_request']
    total_requests_counter = PROMETHEUS_METRICS['total_requests_counter']
    succeed_requests_counter = PROMETHEUS_METRICS['succeed_requests_counter']


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
