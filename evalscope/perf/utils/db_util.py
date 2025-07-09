import base64
import json
import os
import pickle
import re
import sqlite3
import sys
from datetime import datetime
from tabulate import tabulate
from typing import Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.utils.benchmark_util import BenchmarkData, BenchmarkMetrics
from evalscope.utils.logger import get_logger

logger = get_logger()


def encode_data(data) -> str:
    """Encodes data using base64 and pickle."""
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')


def write_json_file(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def transpose_results(data):
    headers = data.keys()
    rows = zip(*data.values())

    return [dict(zip(headers, row)) for row in rows]


def create_result_table(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS result(
                      request TEXT,
                      start_time REAL,
                      chunk_times TEXT,
                      success INTEGER,
                      response_messages TEXT,
                      completed_time REAL,
                      latency REAL,
                      first_chunk_latency REAL,
                      n_chunks INTEGER,
                      chunk_time REAL,
                      prompt_tokens INTEGER,
                      completion_tokens INTEGER,
                      max_gpu_memory_cost REAL)''')


def insert_benchmark_data(cursor: sqlite3.Cursor, benchmark_data: BenchmarkData):
    request = encode_data(benchmark_data.request)
    chunk_times = json.dumps(benchmark_data.chunk_times)
    response_messages = encode_data(benchmark_data.response_messages)

    # Columns common to both success and failure cases
    common_columns = (
        request,
        benchmark_data.start_time,
        chunk_times,
        benchmark_data.success,
        response_messages,
        benchmark_data.completed_time,
    )

    if benchmark_data.success:
        # Add additional columns for success case
        additional_columns = (
            benchmark_data.query_latency,
            benchmark_data.first_chunk_latency,
            benchmark_data.n_chunks,
            benchmark_data.n_chunks_time,
            benchmark_data.prompt_tokens,
            benchmark_data.completion_tokens,
            benchmark_data.max_gpu_memory_cost,
        )
        query = """INSERT INTO result(
                      request, start_time, chunk_times, success, response_messages,
                      completed_time, latency, first_chunk_latency,
                      n_chunks, chunk_time, prompt_tokens, completion_tokens, max_gpu_memory_cost
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns + additional_columns)
    else:
        query = """INSERT INTO result(
                      request, start_time, chunk_times, success, response_messages, completed_time
                   ) VALUES (?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, common_columns)


def get_output_path(args: Arguments) -> str:
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.outputs_dir, current_time, f'{args.name or args.model_id}')
    # Filter illegal characters
    output_path = re.sub(r'[<>:"|?*]', '_', output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    logger.info(f'Save the result to: {output_path}')
    return output_path


def get_result_db_path(args: Arguments):
    result_db_path = os.path.join(args.outputs_dir, 'benchmark_data.db')

    logger.info(f'Save the data base to: {result_db_path}')
    if os.path.exists(result_db_path):
        logger.warning('The db file exists, delete it and start again!.')
        sys.exit(1)

    return result_db_path


class PercentileMetrics:
    TTFT = 'TTFT (s)'
    ITL = 'ITL (s)'
    TPOT = 'TPOT (s)'
    LATENCY = 'Latency (s)'
    INPUT_TOKENS = 'Input tokens'
    OUTPUT_TOKENS = 'Output tokens'
    OUTPUT_THROUGHPUT = 'Output (tok/s)'
    TOTAL_THROUGHPUT = 'Total (tok/s)'
    PERCENTILES = 'Percentiles'


def calculate_percentiles(data: List[float], percentiles: List[int]) -> Dict[int, float]:
    """
    Calculate the percentiles for a specific list of data.

    :param data: List of values for a specific metric.
    :param percentiles: List of percentiles to calculate.
    :return: Dictionary of calculated percentiles.
    """
    results = {}
    n_success_queries = len(data)
    data.sort()
    for percentile in percentiles:
        try:
            idx = int(n_success_queries * percentile / 100)
            value = data[idx] if data[idx] is not None else float('nan')
            results[percentile] = round(value, 4)
        except IndexError:
            results[percentile] = float('nan')
    return results


def get_percentile_results(result_db_path: str) -> Dict[str, List[float]]:
    """
    Compute and return quantiles for various metrics from the database results.

    :param result_db_path: Path to the SQLite database file.
    :return: Dictionary of percentiles for various metrics.
    """

    def inter_token_latencies(chunk_times_json: str) -> List[float]:
        try:
            chunk_times = json.loads(chunk_times_json)
            return [t2 - t1 for t1, t2 in zip(chunk_times[:-1], chunk_times[1:])]
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f'Error parsing chunk times: {e}')
            return []

    query_sql = ('SELECT start_time, chunk_times, success, completed_time, latency, first_chunk_latency, '
                 'n_chunks, chunk_time, prompt_tokens, completion_tokens '
                 'FROM result WHERE success=1')

    percentiles = [10, 25, 50, 66, 75, 80, 90, 95, 98, 99]

    with sqlite3.connect(result_db_path) as con:
        rows = con.execute(query_sql).fetchall()

    # Define index variables for columns
    CHUNK_TIMES_INDEX = 1
    LATENCY_INDEX = 4
    FIRST_CHUNK_LATENCY_INDEX = 5
    CHUNK_TIME_INDEX = 7
    PROMPT_TOKENS_INDEX = 8
    COMPLETION_TOKENS_INDEX = 9

    # Prepare data for each metric
    inter_token_latencies_all = []
    for row in rows:
        inter_token_latencies_all.extend(inter_token_latencies(row[CHUNK_TIMES_INDEX]))

    metrics = {
        PercentileMetrics.TTFT: [row[FIRST_CHUNK_LATENCY_INDEX] for row in rows],
        PercentileMetrics.ITL:
        inter_token_latencies_all,
        PercentileMetrics.TPOT:
        [(row[CHUNK_TIME_INDEX] / row[COMPLETION_TOKENS_INDEX]) if row[COMPLETION_TOKENS_INDEX] > 0 else float('nan')
         for row in rows],
        PercentileMetrics.LATENCY: [row[LATENCY_INDEX] for row in rows],
        PercentileMetrics.INPUT_TOKENS: [row[PROMPT_TOKENS_INDEX] for row in rows],
        PercentileMetrics.OUTPUT_TOKENS: [row[COMPLETION_TOKENS_INDEX] for row in rows],
        PercentileMetrics.OUTPUT_THROUGHPUT:
        [(row[COMPLETION_TOKENS_INDEX] / row[LATENCY_INDEX]) if row[LATENCY_INDEX] > 0 else float('nan')
         for row in rows],
        PercentileMetrics.TOTAL_THROUGHPUT: [((row[PROMPT_TOKENS_INDEX] + row[COMPLETION_TOKENS_INDEX])
                                              / row[LATENCY_INDEX]) if row[LATENCY_INDEX] > 0 else float('nan')
                                             for row in rows]
    }

    # Calculate percentiles for each metric
    results = {PercentileMetrics.PERCENTILES: [f'{p}%' for p in percentiles]}
    for metric_name, data in metrics.items():
        metric_percentiles = calculate_percentiles(data, percentiles)
        results[metric_name] = [metric_percentiles[p] for p in percentiles]

    return results


def summary_result(args: Arguments, metrics: BenchmarkMetrics, result_db_path: str) -> Tuple[Dict, Dict]:
    result_path = os.path.dirname(result_db_path)
    write_json_file(args.to_dict(), os.path.join(result_path, 'benchmark_args.json'))

    metrics_result = metrics.create_message()
    write_json_file(metrics_result, os.path.join(result_path, 'benchmark_summary.json'))

    # Print summary in a table
    table = tabulate(list(metrics_result.items()), headers=['Key', 'Value'], tablefmt='grid')
    logger.info('\nBenchmarking summary:\n' + table)

    # Get percentile results
    percentile_result = get_percentile_results(result_db_path)
    if percentile_result:
        write_json_file(transpose_results(percentile_result), os.path.join(result_path, 'benchmark_percentile.json'))
        # Print percentile results in a table
        table = tabulate(percentile_result, headers='keys', tablefmt='pretty')
        logger.info('\nPercentile results:\n' + table)

    if args.dataset.startswith('speed_benchmark'):
        speed_benchmark_result(result_db_path)

    logger.info(f'Save the summary to: {result_path}')

    return metrics_result, percentile_result


def speed_benchmark_result(result_db_path: str):
    query_sql = """
        SELECT
            prompt_tokens,
            ROUND(AVG(completion_tokens / latency), 2) AS avg_completion_token_per_second,
            ROUND(AVG(max_gpu_memory_cost), 2)
        FROM
            result
        WHERE
            success = 1 AND latency > 0
        GROUP BY
            prompt_tokens
    """

    with sqlite3.connect(result_db_path) as con:
        cursor = con.cursor()
        cursor.execute(query_sql)
        rows = cursor.fetchall()

    # Prepare data for tabulation
    headers = ['Prompt Tokens', 'Speed(tokens/s)', 'GPU Memory(GB)']
    data = [dict(zip(headers, row)) for row in rows]

    # Print results in a table
    table = tabulate(data, headers='keys', tablefmt='pretty')
    logger.info('\nSpeed Benchmark Results:\n' + table)

    # Write results to JSON file
    result_path = os.path.dirname(result_db_path)
    write_json_file(data, os.path.join(result_path, 'speed_benchmark.json'))


# MySQL specific functions
def _flatten_dict(data: dict, parent_key: str = '', sep: str = '_') -> dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_mysql_connection(args: Arguments):
    """Establishes a connection to the MySQL database."""
    try:
        import mysql.connector
        from mysql.connector import errorcode
    except ImportError:
        logger.error("mysql-connector-python is not installed. Please install it via 'pip install mysql-connector-python'")
        return None

    try:
        connection = mysql.connector.connect(
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password,
            database=args.db_name
        )
        return connection
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            logger.error("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            logger.error(f"Database '{args.db_name}' does not exist")
        else:
            logger.error(f"Failed to connect to MySQL: {err}")
        return None


def _get_sql_type(value) -> str:
    """Infers SQL type from a Python value."""
    if isinstance(value, int):
        return 'INT'
    elif isinstance(value, float):
        return 'FLOAT'
    elif isinstance(value, str):
        return 'VARCHAR(255)'
    else:
        return 'TEXT'


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

def save_to_sql(args: Arguments, metrics_result: dict, percentile_result: dict, start_timestamp: float, end_timestamp: float):
    base_labels = {
        'model': args.model_id,
        'concurrency': str(args.parallel),
        'dataset': args.dataset,
        'ep': str(args.ep) if args.ep is not None else 'unkown',
        'dp': str(args.dp) if args.dp is not None else 'unkown',
        'tp': str(args.tp) if args.tp is not None else 'unkown',
        'pd': args.pd if args.pd is not None else 'unkown',
        'metadata': args.metadata if args.metadata is not None else 'unkown',
        'engine': args.engine if args.engine is not None else 'unkown',
        'ISL': args.max_prompt_length, 
        'OSL': args.min_tokens,
        'ignore_eos': args.extra_args['ignore_eos'] if (args.extra_args and 'ignore_eos' in args.extra_args) else 'False',
    }
    flat_percentiles = _restructure_percentiles(percentile_result)
    db_metrics = {
        **base_labels,
        **metrics_result,
        **flat_percentiles,  # Merge the flattened percentile data
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp
    }
    save_to_mysql(args, db_metrics)
    
def _get_existing_columns(cursor, table_name: str) -> set:
    """Fetches the set of existing column names for a given table."""
    try:
        cursor.execute(f"DESCRIBE `{table_name}`")
        return {row[0] for row in cursor.fetchall()}
    except Exception as e:
        # This can happen if the table doesn't exist yet, which is fine.
        if "doesn't exist" in str(e):
            return set()
        raise e

def save_to_mysql(args: Arguments, data: dict):
    """
    Saves benchmark data to a fixed table in a MySQL database,
    dynamically adding new columns as needed.
    """
    # Use a fixed table name from args instead of a dynamic one
    table_name = args.db_table_name
    
    # Add metadata to the data dictionary to be inserted as a column
    if args.metadata:
        data['metadata'] = args.metadata

    logger.info(f"Attempting to save results to MySQL table: `{table_name}`")

    connection = _get_mysql_connection(args)
    if not connection:
        return

    try:
        cursor = connection.cursor()

        # Flatten the data for a flat table structure
        flat_data = _flatten_dict(data)
        
        # Get existing columns and find new ones
        existing_columns = _get_existing_columns(cursor, table_name)
        new_columns = set(flat_data.keys()) - existing_columns

        # If the table doesn't exist, create it
        if not existing_columns:
            columns_defs = [f"`{col}` {_get_sql_type(val)}" for col, val in flat_data.items()]
            create_table_sql = f"CREATE TABLE `{table_name}` ({', '.join(columns_defs)})"
            logger.debug(f"Table `{table_name}` not found. Executing CREATE TABLE: {create_table_sql}")
            cursor.execute(create_table_sql)
        # If there are new columns, add them to the existing table
        elif new_columns:
            for col in new_columns:
                col_type = _get_sql_type(flat_data[col])
                alter_table_sql = f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` {col_type}"
                logger.debug(f"Found new column. Executing ALTER TABLE: {alter_table_sql}")
                cursor.execute(alter_table_sql)

        # Dynamically create the INSERT statement
        columns = '`, `'.join(flat_data.keys())
        placeholders = ', '.join(['%s'] * len(flat_data))
        insert_sql = f"INSERT INTO `{table_name}` (`{columns}`) VALUES ({placeholders})"
        logger.debug(f"Executing INSERT: {insert_sql}")
        cursor.execute(insert_sql, list(flat_data.values()))

        connection.commit()
        logger.info(f"Successfully saved results to MySQL table: `{table_name}` in database `{args.db_name}`")

    except Exception as e:
        logger.error(f"Failed to save data to MySQL: {e}")
        if connection.is_connected():
            connection.rollback()
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
