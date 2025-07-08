import os

from evalscope.perf.arguments import Arguments


def init_wandb(args: Arguments) -> None:
    """
    Initialize WandB for logging.
    """
    # Initialize wandb if the api key is provided
    import datetime
    try:
        import wandb
    except ImportError:
        raise RuntimeError('Cannot import wandb. Please install it with command: \n pip install wandb')
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_DIR'] = args.outputs_dir

    wandb.login(key=args.wandb_api_key)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = args.name if args.name else f'{args.model_id}_{current_time}'
    wandb.init(project='perf_benchmark', name=name, config=args.to_dict())


def init_swanlab(args: Arguments) -> None:
    import datetime
    try:
        import swanlab
    except ImportError:
        raise RuntimeError('Cannot import swanlab. Please install it with command: \n pip install swanlab')
    os.environ['SWANLAB_LOG_DIR'] = args.outputs_dir
    if not args.swanlab_api_key == 'local':
        swanlab.login(api_key=args.swanlab_api_key, host=args.swanlab_host, web_host=args.swanlab_web_host, save=args.swanlab_save)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = args.name if args.name else f'{args.model_id}_{current_time}'
    swanlab.config.update({'framework': '📏evalscope'})
    init_kwargs = {
        'project': os.getenv('SWANLAB_PROJ_NAME', 'perf_benchmark'),
        'name': name,
        'config': args.to_dict(),
        'mode': 'local' if args.swanlab_api_key == 'local' else None
    }

    workspace = os.getenv('SWANLAB_WORKSPACE')
    if workspace:
        init_kwargs['workspace'] = workspace

    swanlab.init(**init_kwargs)


def init_prometheus(args: Arguments):
    try:
        from prometheus_client import CollectorRegistry, Counter, Gauge
    except ImportError:
        raise RuntimeError('Cannot import prometheus_client. Please install it with command: \n pip install prometheus_client')

    registry = CollectorRegistry()
    base_labels = ['model', 'concurrency', 'dataset', 'ep', 'dp', 'tp', 'pd', 'metadata']

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


    return {
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
        # NEW metrics
        'avg_time_per_output_token': avg_time_per_output_token,
        'avg_package_latency': avg_package_latency,
        'avg_package_per_request': avg_package_per_request,
        'total_requests_counter': total_requests_counter,
        'succeed_requests_counter': succeed_requests_counter,
    }