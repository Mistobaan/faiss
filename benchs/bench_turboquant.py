# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""bench_turboquant.py is a quantization benchmark harness inspired by bench_fw.

This script benchmarks codec quality and search quality on a small set of
HuggingFace-backed datasets:

- `sift-128-1M`
- `glove-100-6B`
- `glove-200-6B`
- `dbpedia-1536-1M`
- `dbpedia-3072-1M`

Main outputs:

- bench_fw-compatible `result.json`
- optional row-level debug `jsonl`
- static Pareto/error/training-time plots

Main workflows:

1. Download a dataset into the local Hugging Face cache:

   `python3 benchs/bench_turboquant.py --download glove-100-6B`

2. Run codec and search benchmarks, write `result.json`, and render plots:

   `python3 benchs/bench_turboquant.py --benchmark-mode both --num-samples 100000 --query-count 1000 --result-json benchs/out/glove100.json --plot-dir benchs/out/glove100_plots glove-100-6B PQ32x8 OPQ32x8 TQMSE8`

3. Run a search-only sweep on SIFT with recomputed GT for a truncated corpus:

   `python3 benchs/bench_turboquant.py --benchmark-mode search --num-samples 200000 --query-count 1000 --nprobe-values 4,16,32 --result-json benchs/out/sift_search.json --plot-dir benchs/out/sift_search_plots sift-128-1M PQ32x8 OPQ32x8 RaBitQ8`

4. Run codec-only comparisons:

   `python3 benchs/bench_turboquant.py --benchmark-mode codec --num-samples 100000 --query-count 1000 --result-json benchs/out/dbpedia_codec.json dbpedia-1536-1M PQ96x8 RQ32x8 TQProd8`

5. Regenerate plots from an existing benchmark result without rerunning:

   `python3 benchs/bench_turboquant.py --plot-only --result-json benchs/out/glove100.json --plot-dir benchs/out/glove100_plots`

Notes:

- `--num-samples` always means corpus size.
- `--query-count` limits the evaluated query set.
- `--output-jsonl` is optional and intended for debugging, not plotting.
- `--no-plots` skips plot generation after the run.
"""

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
import importlib.machinery
import importlib.util
import itertools
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
from faiss.contrib.vecs_io import fvecs_read, ivecs_read
import numpy as np
from huggingface_hub import hf_hub_download
from bench_fw.utils import ParetoMetric, ParetoMode, filter_results


def import_huggingface_datasets_module():
    local_datasets_path = (Path(__file__).resolve().parent / "datasets.py").resolve()

    existing = sys.modules.get("datasets")
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is None or Path(existing_path).resolve() != local_datasets_path:
            return existing
        del sys.modules["datasets"]

    search_paths = []
    seen = set()
    for entry in sys.path:
        resolved = Path(entry or Path.cwd()).resolve()
        if resolved == local_datasets_path.parent:
            continue
        resolved_str = str(resolved)
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        search_paths.append(resolved_str)

    spec = importlib.machinery.PathFinder.find_spec("datasets", search_paths)
    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not find the Hugging Face 'datasets' package outside "
            f"{local_datasets_path}. Install the external 'datasets' package "
            "in the Python environment that runs this script."
        )

    origin = getattr(spec, "origin", None)
    if origin is not None and Path(origin).resolve() == local_datasets_path:
        raise ImportError(
            f"Refusing to import local module {local_datasets_path} as Hugging Face "
            "'datasets'."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_hf_datasets = import_huggingface_datasets_module()
DownloadConfig = _hf_datasets.DownloadConfig
DownloadManager = _hf_datasets.DownloadManager
load_dataset = _hf_datasets.load_dataset


def normalize_dataset_metric_name(metric):
    metric_name = str(metric).strip().upper()
    if metric_name == "L2":
        return "L2"
    if metric_name == "IP":
        return "IP"
    raise ValueError(f"unsupported metric {metric}")


class Dataset(ABC):
    """Common interface for benchmark datasets."""

    def __init__(
        self,
        *,
        d: int,
        nq: int,
        nb: int,
        nt: int,
        metric: str = "L2",
    ) -> None:
        self.d = d
        self.metric = normalize_dataset_metric_name(metric)
        self.nq = nq
        self.nb = nb
        self.nt = nt

    @abstractmethod
    def get_queries(self) -> np.ndarray:
        """Return query vectors."""

    @abstractmethod
    def get_train(self, maxtrain: Optional[int] = None) -> np.ndarray:
        """Return training vectors."""

    @abstractmethod
    def get_database(self) -> np.ndarray:
        """Return database vectors."""

    def database_iterator(
        self,
        bs: int = 128,
        split: tuple[int, int] = (1, 0),
    ) -> Iterator[np.ndarray]:
        if bs <= 0:
            raise ValueError("batch size must be positive")
        nsplit, rank = split
        if nsplit <= 0:
            raise ValueError("nsplit must be positive")
        if not 0 <= rank < nsplit:
            raise ValueError("rank must satisfy 0 <= rank < nsplit")
        xb = self.get_database()
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        for j0 in range(i0, i1, bs):
            yield xb[j0:min(j0 + bs, i1)]

    @abstractmethod
    def get_groundtruth(self, k: Optional[int] = None) -> np.ndarray:
        """Return ground-truth neighbor ids."""


class DatasetSIFT1M(Dataset):
    dimension = 128
    dataset_label = "SIFT 128 1M"
    reference_name = "sift-128-1M"
    repo_id = "qbo-odp/sift1m"
    file_names = (
        "sift_learn.fvecs",
        "sift_base.fvecs",
        "sift_query.fvecs",
        "sift_groundtruth.ivecs",
    )

    def __init__(self):
        super().__init__(d=self.dimension, nq=10000, nb=1000000, nt=100000, metric="L2")
        self.file_paths = None

    @classmethod
    def download(cls):
        download_manager = DownloadManager(
            dataset_name=cls.repo_id,
            download_config=DownloadConfig(local_files_only=False),
        )
        return download_manager.download(
            {
                name: f"https://huggingface.co/datasets/{cls.repo_id}/resolve/main/{name}"
                for name in cls.file_names
            }
        )

    def get_file_paths(self, local_only=True):
        if self.file_paths is None or not local_only:
            download_manager = DownloadManager(
                dataset_name=self.repo_id,
                download_config=DownloadConfig(local_files_only=local_only),
            )
            try:
                self.file_paths = download_manager.download(
                    {
                        name: (
                            f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{name}"
                        )
                        for name in self.file_names
                    }
                )
            except Exception as exc:
                if local_only:
                    raise FileNotFoundError(
                        f"{self.dataset_label} is not available in the local Hugging Face "
                        f"cache. Call {self.__class__.__name__}.download() first."
                    ) from exc
                raise
        return self.file_paths

    def get_queries(self):
        return fvecs_read(self.get_file_paths()["sift_query.fvecs"])

    def get_train(self, maxtrain=None):
        maxtrain = maxtrain if maxtrain is not None else self.nt
        return fvecs_read(self.get_file_paths()["sift_learn.fvecs"])[:maxtrain]

    def get_database(self):
        return fvecs_read(self.get_file_paths()["sift_base.fvecs"])

    def get_groundtruth(self, k=None):
        gt = ivecs_read(self.get_file_paths()["sift_groundtruth.ivecs"])
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt


class DatasetSelfRetrievalEmbeddings(Dataset, ABC):
    def __init__(self, d: int, metric: str = "IP"):
        super().__init__(d=d, nq=0, nb=0, nt=0, metric=metric)
        self.database = None
        self.queries = None

    @abstractmethod
    def load_embeddings(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> np.ndarray:
        """Load embeddings for the configured range."""

    def get_queries(self):
        if self.queries is None:
            if self.database is not None:
                self.queries = np.ascontiguousarray(
                    self.database[self.query_offset:self.nb]
                )
            else:
                self.queries = self.load_embeddings(self.query_offset, self.nb)
        return self.queries

    def get_database(self):
        if self.database is None:
            self.database = self.load_embeddings()
        return self.database

    def get_groundtruth(self, k=None):
        gt = -np.ones((self.nq, 1), dtype="int64")
        gt[:, 0] = np.arange(self.query_offset, self.nb, dtype="int64")
        return gt

    def get_train(self, maxtrain=None):
        database = self.get_database()
        available_train_vectors = self.query_offset if self.query_offset > 0 else self.nb
        if maxtrain is not None:
            available_train_vectors = min(available_train_vectors, maxtrain)
        return np.ascontiguousarray(database[:available_train_vectors])


class DatasetDBpediaTextEmbeddings(DatasetSelfRetrievalEmbeddings):
    repo_id = None
    embedding_column = None
    dimension = None
    default_nb = None

    def __init__(
        self,
        num_samples=None,
        num_queries=100,
        metric="IP",
    ):
        super().__init__(d=self.dimension, metric=metric)
        if num_samples is not None and num_samples <= 0:
            raise ValueError("--num-samples must be positive")
        if num_queries <= 0:
            raise ValueError("--query-count must be positive")

        self.num_samples = self.default_nb if num_samples is None else int(num_samples)
        self.dataset = None

        ds = self.get_dataset()
        self.nb = min(self.num_samples, len(ds))
        self.nq = min(int(num_queries), self.nb)
        self.query_offset = self.nb - self.nq
        self.nt = self.query_offset if self.query_offset > 0 else self.nb

    @classmethod
    def load_remote_dataset(cls, local_only):
        try:
            return load_dataset(
                cls.repo_id,
                split="train",
                download_config=DownloadConfig(local_files_only=local_only),
            ).with_format(
                "numpy",
                columns=[cls.embedding_column],
                output_all_columns=False,
            )
        except Exception as exc:
            if local_only:
                raise FileNotFoundError(
                    "DBpedia text embeddings are not available in the local Hugging Face "
                    f"cache. Call {cls.__name__}.download() first."
                ) from exc
            raise

    @classmethod
    def download(cls):
        cls.load_remote_dataset(local_only=False)

    def get_dataset(self):
        if self.dataset is None:
            self.dataset = self.load_remote_dataset(local_only=True)
        return self.dataset

    def load_embeddings(self, start=None, stop=None):
        ds = self.get_dataset()
        if start is None:
            start = 0
        if stop is None:
            stop = self.nb
        return np.asarray(ds[start:stop][self.embedding_column], dtype="float32")


class DatasetDBpedia1536_1M(DatasetDBpediaTextEmbeddings):
    repo_id = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
    embedding_column = "text-embedding-3-large-1536-embedding"
    dimension = 1536
    default_nb = 10**6


class DatasetDBpedia3072_1M(DatasetDBpediaTextEmbeddings):
    repo_id = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
    embedding_column = "text-embedding-3-large-3072-embedding"
    dimension = 3072
    default_nb = 10**6


class DatasetGlove6B(DatasetSelfRetrievalEmbeddings):
    repo_id = "stanfordnlp/glove"
    archive_name = "glove.6B.zip"
    dataset_label = None
    reference_name = None
    embedding_name = None
    dimension = None
    default_nb = 400000

    def __init__(
        self,
        num_samples=None,
        num_queries=100,
        metric="IP",
    ):
        super().__init__(d=self.dimension, metric=metric)
        if num_samples is not None and num_samples <= 0:
            raise ValueError("--num-samples must be positive")
        if num_queries <= 0:
            raise ValueError("--query-count must be positive")

        self.num_samples = self.default_nb if num_samples is None else int(num_samples)
        self.dataset = None
        self.database = self.load_embeddings()
        self.nb = self.database.shape[0]
        self.nq = min(int(num_queries), self.nb)
        self.query_offset = self.nb - self.nq
        self.nt = self.query_offset if self.query_offset > 0 else self.nb

    @classmethod
    def load_text_dataset(cls, local_only=True):
        try:
            archive_path = hf_hub_download(
                repo_id=cls.repo_id,
                filename=cls.archive_name,
                local_files_only=local_only,
            )
            return load_dataset(
                "text",
                data_files={"train": f"zip://{cls.embedding_name}::{archive_path}"},
                split="train",
                download_config=DownloadConfig(local_files_only=local_only),
            )
        except Exception as exc:
            if local_only:
                raise FileNotFoundError(
                    f"{cls.dataset_label} is not available in the local Hugging Face "
                    f"cache. Call {cls.__name__}.download() first."
                ) from exc
            raise

    @classmethod
    def download(cls):
        cls.load_text_dataset(local_only=False)

    def get_text_dataset(self):
        if self.dataset is None:
            self.dataset = self.load_text_dataset(local_only=True)
        return self.dataset

    @staticmethod
    def load_glove_lines(
        lines: Iterable[str],
        max_vectors: Optional[int] = None,
        expected_dim: Optional[int] = None,
    ):
        vectors = []

        for line in lines:
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue

            vals = parts[1:]

            if expected_dim is not None and len(vals) != expected_dim:
                continue

            try:
                vec = np.asarray(vals, dtype=np.float32)
            except ValueError:
                continue

            if expected_dim is None:
                expected_dim = vec.shape[0]

            if vec.shape[0] != expected_dim:
                continue

            vectors.append(vec)

            if max_vectors is not None and len(vectors) >= max_vectors:
                break

        if not vectors:
            raise ValueError("No vectors loaded from GloVe input")

        return np.vstack(vectors).astype(np.float32, copy=False)

    def load_embeddings(self, start=None, stop=None):
        if self.database is None:
            ds = self.get_text_dataset()
            self.database = self.load_glove_lines(
                ds["text"],
                max_vectors=self.num_samples,
                expected_dim=self.d,
            )

        if start is None and stop is None:
            return self.database
        return np.ascontiguousarray(self.database[start:stop])


class DatasetGlove100_6B(DatasetGlove6B):
    dataset_label = "GloVe 100 6B"
    reference_name = "glove-100-6B"
    embedding_name = "glove.6B.100d.txt"
    dimension = 100


class DatasetGlove200_6B(DatasetGlove6B):
    dataset_label = "GloVe 200 6B"
    reference_name = "glove-200-6B"
    embedding_name = "glove.6B.200d.txt"
    dimension = 200


DATASET_NAMES = (
    DatasetSIFT1M.reference_name,
    DatasetGlove100_6B.reference_name,
    DatasetGlove200_6B.reference_name,
    "dbpedia-1536-1M",
    "dbpedia-3072-1M",
)
DEFAULT_RECALL_AT = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_NPROBE_VALUES = (4, 16, 32)
DOWNLOADABLE_DATASET_NAMES = {
    DatasetSIFT1M.reference_name,
    DatasetGlove100_6B.reference_name,
    DatasetGlove200_6B.reference_name,
    "dbpedia-1536-1M",
    "dbpedia-3072-1M",
}
SELF_RETRIEVAL_DATASET_NAMES = {
    DatasetGlove100_6B.reference_name,
    DatasetGlove200_6B.reference_name,
    "dbpedia-1536-1M",
    "dbpedia-3072-1M",
}
DATASET_METRICS = {
    DatasetSIFT1M.reference_name: "l2",
    DatasetGlove100_6B.reference_name: "ip",
    DatasetGlove200_6B.reference_name: "ip",
    "dbpedia-1536-1M": "ip",
    "dbpedia-3072-1M": "ip",
}
DEFAULT_SELF_RETRIEVAL_NQ = 100
AQ_SEARCH_TYPE_CHOICES = {
    "decompress": "ST_decompress",
    "norm_float": "ST_norm_float",
    "norm_qint8": "ST_norm_qint8",
    "norm_qint4": "ST_norm_qint4",
    "norm_cqint8": "ST_norm_cqint8",
    "norm_cqint4": "ST_norm_cqint4",
}
RESULT_REPORTER = None


@dataclass(frozen=True)
class IndexSpec:
    family: str
    label: str
    M: int | None = None
    nbits: int | None = None
    nsplits: int | None = None
    Msub: int | None = None


def normalize_name(value):
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(value))
    value = re.sub(r"_+", "_", value).strip("_")
    return value.lower()


def normalize_metric_name(value):
    return normalize_name(value)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, np.floating):
        return f"{float(value):.6f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def now_ns():
    return time.perf_counter_ns()


def elapsed_ms(start_ns, end_ns=None):
    if end_ns is None:
        end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1e6


def json_value(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, np.floating):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def format_fields(fields):
    return " ".join(f"{key}={format_value(val)}" for key, val in fields)


def emit_line(fields, stream=sys.stdout):
    print(format_fields(fields), file=stream, flush=True)


def emit_progress(fields):
    emit_line(fields, stream=sys.stderr)


def emit_result(fields):
    emit_line(fields)
    if RESULT_REPORTER is not None:
        RESULT_REPORTER.record_result(fields)


def fields_to_row(fields):
    return {key: json_value(value) for key, value in fields}


def default_output_path(run_id, extension):
    return Path.cwd() / f"benchmark-{run_id}.{extension}"


def default_plot_dir(result_json_path):
    result_json_path = Path(result_json_path)
    return result_json_path.parent / f"{result_json_path.stem}_plots"


def find_recall_value(row):
    if "recall_at_1" in row:
        return row["recall_at_1"]
    recall_fields = sorted(
        (
            (int(key.split("_")[-1]), value)
            for key, value in row.items()
            if key.startswith("recall_at_")
        ),
        key=lambda item: item[0],
    )
    if not recall_fields:
        return None
    return recall_fields[0][1]


def collect_search_params(row):
    params = {}
    for key in ("nprobe", "qb", "centered"):
        if key in row:
            params[key] = row[key]
    return params


def value_in_seconds(row, key):
    if key not in row:
        return None
    return float(row[key]) / 1000.0


def code_size_from_row(row):
    if "code_size_bytes" in row:
        return row["code_size_bytes"]
    if "bytes_per_code" in row:
        return row["bytes_per_code"]
    if "memory_bytes" in row and row.get("database_size"):
        database_size = row["database_size"]
        if database_size > 0:
            return row["memory_bytes"] / float(database_size)
    return None


class ResultReporter:
    def __init__(self, result_json_path, output_jsonl_path, total_jobs, total_results):
        self.result_json_path = Path(result_json_path)
        self.result_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_jsonl_path = Path(output_jsonl_path) if output_jsonl_path else None
        if self.output_jsonl_path is not None:
            self.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.output_jsonl_path.open("w", encoding="utf-8")
        else:
            self.file = None
        self.results = {"indices": {}, "experiments": {}}
        self.total_jobs = total_jobs
        self.total_results = total_results
        self.completed_jobs = 0
        self.completed_results = 0

        progress_fields = [
            ("progress_phase", "benchmark_start"),
            ("result_json_path", str(self.result_json_path)),
            ("total_jobs", total_jobs),
            ("total_results", total_results),
        ]
        if self.output_jsonl_path is not None:
            progress_fields.append(("output_jsonl_path", str(self.output_jsonl_path)))
        emit_progress(progress_fields)

    def start_job(self, benchmark_mode, job_name, expected_results):
        emit_progress(
            [
                ("progress_phase", "job_start"),
                ("benchmark_mode", normalize_name(benchmark_mode)),
                ("job_name", normalize_name(job_name)),
                ("job_index", self.completed_jobs + 1),
                ("total_jobs", self.total_jobs),
                ("expected_results", expected_results),
                ("completed_results", self.completed_results),
                ("total_results", self.total_results),
            ]
        )

    def finish_job(self, benchmark_mode, job_name):
        self.completed_jobs += 1
        emit_progress(
            [
                ("progress_phase", "job_done"),
                ("benchmark_mode", normalize_name(benchmark_mode)),
                ("job_name", normalize_name(job_name)),
                ("completed_jobs", self.completed_jobs),
                ("total_jobs", self.total_jobs),
                ("completed_results", self.completed_results),
                ("total_results", self.total_results),
            ]
        )

    def update_index_meta(self, row):
        index_name = row.get("index_name")
        if index_name is None:
            return

        meta = self.results["indices"].setdefault(index_name, {})
        code_size = code_size_from_row(row)
        training_time = value_in_seconds(row, "train_time_ms")

        if code_size is not None:
            meta["code_size"] = code_size
        if training_time is not None:
            meta["training_time"] = training_time
        if "train_size" in row:
            meta["training_size"] = row["train_size"]
        if "memory_bytes" in row:
            meta["codec_size"] = row["memory_bytes"]
        if "metric" in row:
            meta["metric"] = row["metric"]
        if "dimension" in row:
            meta["dimension"] = row["dimension"]

    def add_experiment(self, row):
        index_name = row.get("index_name")
        if index_name is None:
            return

        recall = find_recall_value(row)
        if recall is None:
            return

        benchmark_mode = row.get("benchmark_mode")
        search_params = collect_search_params(row)
        factory = row.get("index_name")
        experiment = {
            "index": index_name,
            "codec": index_name,
            "factory": factory,
        }

        if benchmark_mode == "search":
            search_time = value_in_seconds(row, "search_time_ms")
            if search_time is None:
                return
            search_params = {"snap": 0} | search_params
            experiment |= {
                "time": search_time,
                "k": row["search_k"],
                "knn_intersection": recall,
                "search_params": search_params,
            }
            key = (
                f"{index_name}.knn.search_k_{row['search_k']}"
                f".{len(self.results['experiments'])}"
            )
        elif benchmark_mode == "codec":
            encode_time = value_in_seconds(row, "encode_time_ms")
            if encode_time is None:
                return
            experiment |= {
                "encode_time": encode_time,
                "decode_time": value_in_seconds(row, "decode_time_ms"),
                "mse": row.get("mse"),
                "sym_recall": recall,
                "reconstruct_params": search_params,
            }
            key = f"{index_name}.rec.{len(self.results['experiments'])}"
        else:
            return

        self.results["experiments"][key] = experiment

    def record_result(self, fields):
        self.completed_results += 1
        row = fields_to_row(fields)
        if self.file is not None:
            self.file.write(json.dumps(row, ensure_ascii=True) + "\n")
            self.file.flush()
        self.update_index_meta(row)
        self.add_experiment(row)
        emit_progress(
            [
                ("progress_phase", "result"),
                ("completed_results", self.completed_results),
                ("total_results", self.total_results),
                ("completed_jobs", self.completed_jobs),
                ("total_jobs", self.total_jobs),
                ("index_name", row.get("index_name", "unknown")),
            ]
        )

    def close(self):
        self.result_json_path.write_text(
            json.dumps(self.results, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        progress_fields = [
            ("progress_phase", "benchmark_done"),
            ("completed_jobs", self.completed_jobs),
            ("total_jobs", self.total_jobs),
            ("completed_results", self.completed_results),
            ("total_results", self.total_results),
            ("result_json_path", str(self.result_json_path)),
        ]
        if self.output_jsonl_path is not None:
            progress_fields.append(("output_jsonl_path", str(self.output_jsonl_path)))
        emit_progress(progress_fields)
        if self.file is not None:
            self.file.close()


def count_codec_jobs_and_results(index_specs):
    variant_counts = {
        "lsq-gpu": 1,
        "pq": 1,
        "opq": 1,
        "prq": 6,
        "plsq": 5,
        "rq": 6,
        "rq_lut": 7,
        "lsq": 5,
        "tqmse": 1,
        "tqprod": 1,
        "rabitq": 1,
    }
    job_count = len(index_specs)
    result_count = sum(
        variant_counts[index_spec.family] for index_spec in index_specs
    )
    return job_count, result_count


def count_search_results(search_k_values, recall_at, num_points):
    active_search_ks = sum(
        1
        for search_k in search_k_values
        if any(k <= search_k for k in recall_at)
    )
    return active_search_ks * num_points


def parse_int_list(spec, name):
    if not spec:
        raise RuntimeError(f"expected a non-empty {name} list")
    values = tuple(int(x) for x in spec.split(",") if x)
    if not values:
        raise RuntimeError(f"expected a non-empty {name} list")
    if any(v <= 0 for v in values):
        raise RuntimeError(f"{name} values must be positive")
    return values


def resolve_aq_search_type(name):
    if name not in AQ_SEARCH_TYPE_CHOICES:
        choices = ", ".join(sorted(AQ_SEARCH_TYPE_CHOICES))
        raise RuntimeError(f"--aq-search-type must be one of: {choices}")
    if not hasattr(faiss, "AdditiveQuantizer"):
        raise RuntimeError(
            "AdditiveQuantizer search types are not available in this faiss "
            "Python build."
        )
    return getattr(faiss.AdditiveQuantizer, AQ_SEARCH_TYPE_CHOICES[name])


def make_index_spec(family, M=None, nbits=None, nsplits=None, Msub=None):
    family = family.lower()
    if family == "pq":
        label = f"PQ{M}x{nbits}"
    elif family == "opq":
        label = f"OPQ{M}x{nbits}"
    elif family == "rq":
        label = f"RQ{M}x{nbits}"
    elif family == "rq_lut":
        label = f"RQ_LUT{M}x{nbits}"
    elif family == "lsq":
        label = f"LSQ{M}x{nbits}"
    elif family == "lsq-gpu":
        label = f"LSQ_GPU{M}x{nbits}"
    elif family == "prq":
        label = f"PRQ{nsplits}x{Msub}x{nbits}"
    elif family == "plsq":
        label = f"PLSQ{nsplits}x{Msub}x{nbits}"
    elif family == "tqmse":
        label = f"TurboQuantMSE{nbits}"
    elif family == "tqprod":
        label = f"TurboQuantProd{nbits}"
    elif family == "rabitq":
        label = f"RaBitQ{nbits}"
    else:
        raise RuntimeError(f"unsupported benchmark family {family}")
    return IndexSpec(
        family=family,
        label=label,
        M=M,
        nbits=nbits,
        nsplits=nsplits,
        Msub=Msub,
    )


def parse_index_spec(index_spec):
    index_spec = index_spec.strip()
    explicit_patterns = [
        (r"(?i)^pq(\d+)x(\d+)$", lambda m: make_index_spec("pq", M=int(m.group(1)), nbits=int(m.group(2)))),
        (r"(?i)^opq(\d+)x(\d+)$", lambda m: make_index_spec("opq", M=int(m.group(1)), nbits=int(m.group(2)))),
        (r"(?i)^rq(\d+)x(\d+)$", lambda m: make_index_spec("rq", M=int(m.group(1)), nbits=int(m.group(2)))),
        (r"(?i)^rq[_-]?lut(\d+)x(\d+)$", lambda m: make_index_spec("rq_lut", M=int(m.group(1)), nbits=int(m.group(2)))),
        (r"(?i)^lsq(\d+)x(\d+)$", lambda m: make_index_spec("lsq", M=int(m.group(1)), nbits=int(m.group(2)))),
        (r"(?i)^lsq[_-]?gpu(\d+)x(\d+)$", lambda m: make_index_spec("lsq-gpu", M=int(m.group(1)), nbits=int(m.group(2)))),
        (r"(?i)^prq(\d+)x(\d+)x(\d+)$", lambda m: make_index_spec("prq", nsplits=int(m.group(1)), Msub=int(m.group(2)), nbits=int(m.group(3)))),
        (r"(?i)^plsq(\d+)x(\d+)x(\d+)$", lambda m: make_index_spec("plsq", nsplits=int(m.group(1)), Msub=int(m.group(2)), nbits=int(m.group(3)))),
        (r"(?i)^tqmse(\d+)$", lambda m: make_index_spec("tqmse", nbits=int(m.group(1)))),
        (r"(?i)^tqprod(\d+)$", lambda m: make_index_spec("tqprod", nbits=int(m.group(1)))),
        (r"(?i)^rabitq(\d+)$", lambda m: make_index_spec("rabitq", nbits=int(m.group(1)))),
        (r"(?i)^rbq(\d+)$", lambda m: make_index_spec("rabitq", nbits=int(m.group(1)))),
    ]
    for pattern, build in explicit_patterns:
        match = re.match(pattern, index_spec)
        if match:
            return build(match)
    return None


def parse_index_specs(index_specs):
    parsed_index_specs = []
    ignored = []

    for raw_index_spec in index_specs:
        parsed_index_spec = parse_index_spec(raw_index_spec)
        if parsed_index_spec is not None:
            parsed_index_specs.append(parsed_index_spec)
        else:
            ignored.append(raw_index_spec)

    return parsed_index_specs, ignored


def validate_requested_metric(dataset_name, requested_metric):
    if requested_metric is None:
        return
    if dataset_name in SELF_RETRIEVAL_DATASET_NAMES:
        return
    dataset_metric = DATASET_METRICS[dataset_name]
    if requested_metric != dataset_metric:
        raise RuntimeError(
            f"--metric={requested_metric} does not match dataset metric "
            f"{dataset_metric} for {dataset_name}. Remove --metric or use "
            f"--metric {dataset_metric}."
        )


def index_spec_fields(index_spec):
    fields = [
        ("codec_family", normalize_name(index_spec.family)),
        ("codec_spec", normalize_name(index_spec.label)),
    ]
    if index_spec.nbits is not None:
        fields.append(("bits", index_spec.nbits))
    if index_spec.M is not None:
        fields.append(("m", index_spec.M))
    if index_spec.nsplits is not None:
        fields.append(("nsplits", index_spec.nsplits))
    if index_spec.Msub is not None:
        fields.append(("msub", index_spec.Msub))
    return fields


def validate_index_spec(dimension, index_spec, benchmark_mode):
    if index_spec.family == "pq":
        require_pq_compatible_dimension(dimension, index_spec.M, {"pq"})
    if index_spec.family in {"prq", "plsq"}:
        require_product_split_compatible_dimension(
            dimension,
            index_spec.nsplits,
            {index_spec.family},
        )
    if index_spec.family == "tqmse" and not 1 <= index_spec.nbits <= 8:
        raise RuntimeError("TurboQuantMSE supports nbits in [1, 8]")
    if index_spec.family == "tqprod" and not 1 <= index_spec.nbits <= 9:
        raise RuntimeError("TurboQuantProd supports nbits in [1, 9]")
    if index_spec.family == "rabitq" and not 1 <= index_spec.nbits <= 9:
        raise RuntimeError("RaBitQ supports nbits in [1, 9]")
    if index_spec.family == "lsq-gpu" and benchmark_mode in {"search", "both"}:
        raise RuntimeError("lsq-gpu is only supported in codec benchmark mode")


def valid_divisors(value):
    return [d for d in range(1, value + 1) if value % d == 0]


def require_pq_compatible_dimension(d, M, selected_options):
    if M is None or "pq" not in selected_options:
        return
    if d % M == 0:
        return
    divisors = ",".join(str(v) for v in valid_divisors(d))
    raise RuntimeError(
        f"pq requires dimension % M == 0, but dimension={d} and M={M}. "
        f"Choose an M that divides {d} ({divisors}), or drop pq and use opq "
        "instead."
    )


def require_product_split_compatible_dimension(d, nsplits, selected_options):
    if nsplits is None or not (selected_options & {"prq", "plsq"}):
        return
    if d % nsplits == 0:
        return
    divisors = ",".join(str(v) for v in valid_divisors(d))
    raise RuntimeError(
        f"prq/plsq require dimension % nsplits == 0, but dimension={d} and "
        f"nsplits={nsplits}. Choose an nsplits value that divides {d} "
        f"({divisors})."
    )


def get_metric_type(ds):
    if ds.metric == "IP":
        return faiss.METRIC_INNER_PRODUCT
    if ds.metric == "L2":
        return faiss.METRIC_L2
    raise RuntimeError(f"unsupported dataset metric {ds.metric}")


def configure_omp_threads(num_threads):
    if num_threads is None:
        return None
    if num_threads <= 0:
        raise RuntimeError("--threads must be positive")
    if not hasattr(faiss, "omp_set_num_threads"):
        raise RuntimeError(
            "this faiss Python build does not expose omp_set_num_threads"
        )
    faiss.omp_set_num_threads(num_threads)
    return num_threads


def get_training_vectors(ds, xb, maxtrain):
    try:
        xt = ds.get_train(maxtrain=maxtrain)
        xt = np.ascontiguousarray(xt)
        if xt.shape[0] == 0:
            xt = np.ascontiguousarray(xb[:min(maxtrain, xb.shape[0])])
            return xt, "database"
        return xt, "dataset_train"
    except (AttributeError, NotImplementedError):
        xt = np.ascontiguousarray(xb[:min(maxtrain, xb.shape[0])])
        return xt, "database"


def encode(codec, x):
    if hasattr(codec, "compute_codes") and hasattr(codec, "decode"):
        return codec.compute_codes(x)
    if hasattr(codec, "sa_encode") and hasattr(codec, "sa_decode"):
        return codec.sa_encode(x)
    raise TypeError(f"unsupported codec type {type(codec).__name__}")


def decode(codec, codes):
    if hasattr(codec, "compute_codes") and hasattr(codec, "decode"):
        return codec.decode(codes)
    if hasattr(codec, "sa_encode") and hasattr(codec, "sa_decode"):
        return codec.sa_decode(codes)
    raise TypeError(f"unsupported codec type {type(codec).__name__}")


def get_code_size(codec):
    if hasattr(codec, "code_size"):
        return int(codec.code_size)
    if hasattr(codec, "sa_code_size"):
        return int(codec.sa_code_size())
    return None


def sample_rows(x, size, rng):
    if size is None:
        return np.ascontiguousarray(x), None
    if size <= 0:
        raise RuntimeError("--query-count must be positive")
    size = min(size, x.shape[0])
    if size == x.shape[0]:
        return np.ascontiguousarray(x), np.arange(size)
    idx = rng.choice(x.shape[0], size=size, replace=False)
    return np.ascontiguousarray(x[idx]), idx


def build_exact_groundtruth(xq, xb, metric_type, k):
    k = min(k, xb.shape[0])
    _, gt = faiss.knn(xq, xb, k, metric=metric_type)
    return np.ascontiguousarray(gt, dtype="int64")


def build_eval_data(ds, query_count, num_samples, max_gt_k, metric_type, rng):
    if isinstance(ds, DatasetSIFT1M):
        xb = np.ascontiguousarray(ds.get_database())
        if num_samples is not None:
            if num_samples <= 0:
                raise RuntimeError("--num-samples must be positive")
            xb = np.ascontiguousarray(xb[:min(num_samples, xb.shape[0])])

        xq, query_idx = sample_rows(ds.get_queries(), query_count, rng)
        if xb.shape[0] < ds.nb:
            gt = build_exact_groundtruth(xq, xb, metric_type, max_gt_k)
            return xb, xq, gt, "truncated_database_queries"

        gt = np.asarray(ds.get_groundtruth(k=max_gt_k), dtype="int64")
        if query_idx is not None:
            gt = np.ascontiguousarray(gt[query_idx])
        return xb, xq, gt, "dataset_queries"

    xb = np.ascontiguousarray(ds.get_database())
    xq = np.ascontiguousarray(ds.get_queries())
    gt = np.asarray(ds.get_groundtruth(k=max_gt_k), dtype="int64")
    return xb, xq, gt, "self_retrieval_queries"


def compute_recall_metrics(gt, predicted_I, recall_at):
    metrics = []
    nq = predicted_I.shape[0]
    if nq == 0:
        raise RuntimeError("query set is empty")
    for k in recall_at:
        if k > predicted_I.shape[1]:
            continue
        gt_cols = min(k, gt.shape[1])
        if gt_cols <= 0:
            continue
        gt_k = np.ascontiguousarray(gt[:, :gt_cols], dtype="int64")
        pred_k = np.ascontiguousarray(predicted_I[:, :k], dtype="int64")
        inter = faiss.eval_intersection(gt_k, pred_k)
        recall = inter / float(nq * gt_cols)
        metrics.append((f"recall_at_{k}", recall))
    return metrics


def eval_codec(codec, xq, xb, gt, metric_type, recall_at):
    t0_ns = now_ns()
    codes = encode(codec, xb)
    t1_ns = now_ns()
    xb_decoded = decode(codec, codes)
    t2_ns = now_ns()
    recons_err = ((xb - xb_decoded) ** 2).sum() / xb.shape[0]
    err_compat = np.linalg.norm(xb - xb_decoded, axis=1).mean()
    xq_decoded = decode(codec, encode(codec, xq))

    max_k = max(recall_at)
    if max_k > xb_decoded.shape[0]:
        raise RuntimeError(
            f"max recall_at={max_k} exceeds database size {xb_decoded.shape[0]}"
        )
    _, I = faiss.knn(xq_decoded, xb_decoded, max_k, metric=metric_type)

    metrics = [
        ("encode_time_ms", elapsed_ms(t0_ns, t1_ns)),
        ("decode_time_ms", elapsed_ms(t1_ns, t2_ns)),
        ("reconstruction_error", recons_err),
        ("mse", recons_err),
        ("reconstruction_error_compat", err_compat),
    ]
    metrics.extend(compute_recall_metrics(gt, I, recall_at))
    code_size = get_code_size(codec)
    if code_size is not None:
        metrics.append(("code_size_bytes", code_size))
        metrics.append(("bytes_per_code", code_size))
    return metrics


def eval_quantizer(
    codec,
    index_spec,
    xq,
    xb,
    gt,
    xt,
    metric_type,
    recall_at,
    base_fields,
    variants=None,
    train_time_offset_ms=0.0,
    extra_fields=None,
):
    if variants is None:
        variants = [(None, None)]
    if extra_fields is None:
        extra_fields = []

    if RESULT_REPORTER is not None:
        RESULT_REPORTER.start_job("codec", index_spec.label, len(variants))

    try:
        t0_ns = now_ns()
        codec.train(xt)
        train_time_ms = train_time_offset_ms + elapsed_ms(t0_ns)

        for name, val in variants:
            variant_fields = []
            if name is not None:
                if isinstance(codec, faiss.ProductAdditiveQuantizer):
                    for i in range(codec.nsplits):
                        subq = faiss.downcast_Quantizer(codec.subquantizer(i))
                        getattr(subq, name)
                        setattr(subq, name, val)
                else:
                    getattr(codec, name)
                    setattr(codec, name, val)
                variant_fields.append((normalize_name(name), val))

            metrics = eval_codec(codec, xq, xb, gt, metric_type, recall_at)
            emit_result(
                base_fields +
                [
                    ("benchmark_mode", "codec"),
                    ("codec", normalize_name(index_spec.family)),
                    ("index_name", normalize_name(f"{index_spec.label}_codec")),
                    ("train_time_ms", train_time_ms),
                ] +
                index_spec_fields(index_spec) +
                extra_fields +
                variant_fields +
                metrics
            )
    finally:
        if RESULT_REPORTER is not None:
            RESULT_REPORTER.finish_job("codec", index_spec.label)


def unwrap_base_index(index):
    cur = faiss.downcast_index(index)
    while type(cur).__name__ == "IndexPreTransform":
        cur = faiss.downcast_index(cur.index)
    return cur


def extract_ivf_index(index):
    try:
        return faiss.downcast_index(faiss.extract_index_ivf(index))
    except RuntimeError:
        return None


def make_metric_flat_index(d, metric_type):
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return faiss.IndexFlatIP(d)
    return faiss.IndexFlatL2(d)


def average_time_ms(fn, trials):
    t0_ns = now_ns()
    for _ in range(trials):
        fn()
    return elapsed_ms(t0_ns) / trials


def estimate_code_size_bytes(index):
    cur = unwrap_base_index(index)
    if hasattr(cur, "code_size"):
        return int(cur.code_size)
    if hasattr(cur, "sa_code_size"):
        return int(cur.sa_code_size())
    return None


def estimate_memory_bytes(index):
    cur = unwrap_base_index(index)
    code_size = estimate_code_size_bytes(index)
    if code_size is not None and hasattr(cur, "ntotal"):
        return code_size * int(cur.ntotal)
    return 0


def configure_search_point(index, point):
    ivf = extract_ivf_index(index)
    leaf = unwrap_base_index(index)
    if "nprobe" in point:
        if ivf is None:
            raise RuntimeError("nprobe was requested for a non-IVF index")
        ivf.nprobe = point["nprobe"]
    for attr in ("qb", "centered"):
        if attr in point and hasattr(leaf, attr):
            setattr(leaf, attr, point[attr])


def benchmark_search_index(
    index,
    index_spec,
    index_name,
    index_kind,
    xq,
    gt,
    recall_at,
    search_k_values,
    trials,
    base_fields,
    spec_fields,
    points,
):
    expected_results = count_search_results(search_k_values, recall_at, len(points))
    if RESULT_REPORTER is not None:
        RESULT_REPORTER.start_job("search", index_name, expected_results)
    try:
        code_size_bytes = estimate_code_size_bytes(index)
        memory_bytes = estimate_memory_bytes(index)
        for search_k in search_k_values:
            active_recall_at = tuple(k for k in recall_at if k <= search_k)
            if not active_recall_at:
                continue
            for point in points:
                configure_search_point(index, point)
                _, I = index.search(xq, search_k)
                search_time_ms = average_time_ms(
                    lambda: index.search(xq, search_k),
                    trials,
                )
                metrics = compute_recall_metrics(gt, I, active_recall_at)
                point_fields = [(normalize_name(k), v) for k, v in point.items()]
                size_fields = []
                if code_size_bytes is not None:
                    size_fields.extend(
                        [
                            ("code_size_bytes", code_size_bytes),
                            ("bytes_per_code", code_size_bytes),
                        ]
                    )
                emit_result(
                    base_fields +
                    [
                        ("benchmark_mode", "search"),
                        ("codec", normalize_name(index_spec.family)),
                        ("index_name", normalize_name(index_name)),
                        ("index_kind", normalize_name(index_kind)),
                        ("search_k", search_k),
                        ("trial_count", trials),
                        ("memory_bytes", memory_bytes),
                        ("search_time_ms", search_time_ms),
                        ("search_time_ms_per_query", search_time_ms / xq.shape[0]),
                    ] +
                    size_fields +
                    index_spec_fields(index_spec) +
                    spec_fields +
                    point_fields +
                    metrics
                )
    finally:
        if RESULT_REPORTER is not None:
            RESULT_REPORTER.finish_job("search", index_name)


def train_and_add_index(index, xt, xb):
    t0_ns = now_ns()
    index.train(xt)
    train_time_ms = elapsed_ms(t0_ns)
    t0_ns = now_ns()
    index.add(xb)
    add_time_ms = elapsed_ms(t0_ns)
    return [
        ("train_time_ms", train_time_ms),
        ("add_time_ms", add_time_ms),
    ]


def make_search_points(index_kind, codec_name, args):
    base_point = {}
    if codec_name == "rabitq":
        base_point["qb"] = args.qb
        if index_kind == "flat":
            base_point["centered"] = int(args.centered)
    if index_kind == "ivf":
        return [
            dict(base_point, nprobe=nprobe)
            for nprobe in args.nprobe_values
        ]
    return [base_point]


def build_search_jobs(
    d,
    metric_type,
    index_specs,
    args,
    nlist,
):
    jobs = []
    store_norm = args.dataset_name not in {
        DatasetGlove100_6B.reference_name,
        DatasetGlove200_6B.reference_name,
    }
    aq_search_type = args.aq_search_type
    aq_search_type_name = args.aq_search_type_name

    for request in index_specs:
        if request.family == "pq":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexPQ(
                    d,
                    request.M,
                    request.nbits,
                    metric_type,
                ),
                [],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request: faiss.IndexIVFPQ(
                    make_metric_flat_index(d, metric_type),
                    d,
                    nlist,
                    request.M,
                    request.nbits,
                    metric_type,
                ),
                [("nlist", nlist)],
            ))
        elif request.family == "opq":
            d2 = ((d + request.M - 1) // request.M) * request.M
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request, d2=d2: faiss.IndexPreTransform(
                    faiss.OPQMatrix(d, request.M, d2),
                    faiss.IndexPQ(d2, request.M, request.nbits, metric_type),
                ),
                [("pretransform", "opq"), ("projected_dimension", d2)],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request, d2=d2: faiss.IndexPreTransform(
                    faiss.OPQMatrix(d, request.M, d2),
                    faiss.IndexIVFPQ(
                        make_metric_flat_index(d2, metric_type),
                        d2,
                        nlist,
                        request.M,
                        request.nbits,
                        metric_type,
                    ),
                ),
                [
                    ("pretransform", "opq"),
                    ("projected_dimension", d2),
                    ("nlist", nlist),
                ],
            ))
        elif request.family == "rq":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexResidualQuantizer(
                    d,
                    request.M,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [("max_beam_size", 30), ("aq_search_type", aq_search_type_name)],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request: faiss.IndexIVFResidualQuantizer(
                    make_metric_flat_index(d, metric_type),
                    d,
                    nlist,
                    request.M,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [
                    ("max_beam_size", 30),
                    ("aq_search_type", aq_search_type_name),
                    ("nlist", nlist),
                ],
            ))
        elif request.family == "rq_lut":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexResidualQuantizer(
                    d,
                    request.M,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [
                    ("max_beam_size", 30),
                    ("use_beam_lut", 1),
                    ("aq_search_type", aq_search_type_name),
                ],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request: faiss.IndexIVFResidualQuantizer(
                    make_metric_flat_index(d, metric_type),
                    d,
                    nlist,
                    request.M,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [
                    ("max_beam_size", 30),
                    ("use_beam_lut", 1),
                    ("aq_search_type", aq_search_type_name),
                    ("nlist", nlist),
                ],
            ))
        elif request.family == "lsq":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexLocalSearchQuantizer(
                    d,
                    request.M,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [("encode_ils_iters", 16), ("aq_search_type", aq_search_type_name)],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request: faiss.IndexIVFLocalSearchQuantizer(
                    make_metric_flat_index(d, metric_type),
                    d,
                    nlist,
                    request.M,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [
                    ("encode_ils_iters", 16),
                    ("aq_search_type", aq_search_type_name),
                    ("nlist", nlist),
                ],
            ))
        elif request.family == "prq":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexProductResidualQuantizer(
                    d,
                    request.nsplits,
                    request.Msub,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [("max_beam_size", 32), ("aq_search_type", aq_search_type_name)],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request: faiss.IndexIVFProductResidualQuantizer(
                    make_metric_flat_index(d, metric_type),
                    d,
                    nlist,
                    request.nsplits,
                    request.Msub,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [
                    ("max_beam_size", 32),
                    ("aq_search_type", aq_search_type_name),
                    ("nlist", nlist),
                ],
            ))
        elif request.family == "plsq":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexProductLocalSearchQuantizer(
                    d,
                    request.nsplits,
                    request.Msub,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [("encode_ils_iters", 16), ("aq_search_type", aq_search_type_name)],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request: faiss.IndexIVFProductLocalSearchQuantizer(
                    make_metric_flat_index(d, metric_type),
                    d,
                    nlist,
                    request.nsplits,
                    request.Msub,
                    request.nbits,
                    metric_type,
                    aq_search_type,
                ),
                [
                    ("encode_ils_iters", 16),
                    ("aq_search_type", aq_search_type_name),
                    ("nlist", nlist),
                ],
            ))
        elif request.family == "tqmse":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexTurboQuantMSE(
                    d,
                    request.nbits,
                    metric_type,
                    args.seed,
                    store_norm,
                ),
                [("store_norm", int(store_norm))],
            ))
        elif request.family == "tqprod":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexTurboQuantProd(
                    d,
                    request.nbits,
                    metric_type,
                    args.seed,
                    store_norm,
                ),
                [("store_norm", int(store_norm))],
            ))
        elif request.family == "rabitq":
            jobs.append((
                request,
                normalize_name(f"{request.label}_flat"),
                "flat",
                lambda request=request: faiss.IndexRaBitQ(d, metric_type, request.nbits),
                [("qb", args.qb), ("centered", int(args.centered))],
            ))
            jobs.append((
                request,
                normalize_name(f"{request.label}_ivf"),
                "ivf",
                lambda request=request: faiss.IndexIVFRaBitQ(
                    make_metric_flat_index(d, metric_type),
                    d,
                    nlist,
                    metric_type,
                    True,
                    request.nbits,
                ),
                [("qb", args.qb), ("nlist", nlist)],
            ))
            if not args.no_rabitq_rrot:
                def make_rabitq_rrot_index(request=request):
                    base = faiss.IndexIVFRaBitQ(
                        make_metric_flat_index(d, metric_type),
                        d,
                        nlist,
                        metric_type,
                        True,
                        request.nbits,
                    )
                    rrot = faiss.RandomRotationMatrix(d, d)
                    rrot.init(args.rabitq_rrot_seed)
                    return faiss.IndexPreTransform(rrot, base)

                jobs.append((
                    request,
                    normalize_name(f"{request.label}_ivf_rrot"),
                    "ivf",
                    make_rabitq_rrot_index,
                    [
                        ("qb", args.qb),
                        ("nlist", nlist),
                        ("pretransform", "random_rotation"),
                        ("rotation_seed", args.rabitq_rrot_seed),
                    ],
                ))

    return jobs


def configure_search_index(index, codec_name, args):
    leaf = unwrap_base_index(index)

    if codec_name == "rq":
        leaf.rq.max_beam_size = 30
    elif codec_name == "rq_lut":
        leaf.rq.max_beam_size = 30
        leaf.rq.use_beam_LUT = 1
    elif codec_name == "lsq":
        leaf.lsq.encode_ils_iters = 16
    elif codec_name == "prq":
        leaf.prq.max_beam_size = 32
    elif codec_name == "plsq":
        leaf.plsq.encode_ils_iters = 16
    elif codec_name == "rabitq":
        if hasattr(leaf, "qb"):
            leaf.qb = args.qb
        if hasattr(leaf, "centered"):
            leaf.centered = args.centered


def load_plot_dependencies():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plot generation. Install it and rerun."
        ) from exc
    return plt


def plot_metric(ax, experiments, accuracy_title, cost_title, plot_space=False):
    grouped_accuracy = {}
    grouped_cost = {}

    for accuracy, space, elapsed_time, _, experiment in experiments:
        label = experiment.get("factory", experiment["index"])
        grouped_accuracy.setdefault(label, []).append(accuracy)
        grouped_cost.setdefault(label, []).append(space if plot_space else elapsed_time)

    ax.set_xlabel(accuracy_title)
    ax.set_ylabel(cost_title)
    if not grouped_accuracy:
        ax.text(0.5, 0.5, "No matching experiments", ha="center", va="center")
        return

    marker = itertools.cycle(
        ("o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D")
    )
    all_costs = []
    for label in grouped_accuracy:
        points = sorted(
            zip(grouped_accuracy[label], grouped_cost[label], strict=True),
            key=lambda item: item[0],
        )
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, marker=next(marker), label=label, linewidth=0)
        all_costs.extend(value for value in ys if value is not None)

    if all_costs and all(value > 0 for value in all_costs):
        ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")


def plot_codec_mse_vs_encode_time(ax, results):
    grouped = {}
    for key, experiment in results["experiments"].items():
        if ".rec" not in key:
            continue
        mse = experiment.get("mse")
        encode_time = experiment.get("encode_time")
        if mse is None or encode_time is None:
            continue
        label = experiment.get("factory", experiment["index"])
        grouped.setdefault(label, []).append((mse, encode_time))

    ax.set_xlabel("mse")
    ax.set_ylabel("encode time (seconds)")
    if not grouped:
        ax.text(0.5, 0.5, "No reconstruction experiments", ha="center", va="center")
        return

    marker = itertools.cycle(
        ("o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D")
    )
    x_values = []
    y_values = []
    for label, pairs in grouped.items():
        pairs.sort(key=lambda item: item[0])
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        ax.plot(xs, ys, marker=next(marker), label=label, linewidth=0)
        x_values.extend(xs)
        y_values.extend(ys)

    if x_values and all(value > 0 for value in x_values):
        ax.set_xscale("log")
    if y_values and all(value > 0 for value in y_values):
        ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")


def plot_training_time_vs_recall(ax, results):
    best_recall = {}
    for key, experiment in results["experiments"].items():
        if ".knn" not in key:
            continue
        recall = experiment.get("knn_intersection")
        if recall is None:
            continue
        index_name = experiment["index"]
        best_recall[index_name] = max(best_recall.get(index_name, 0.0), recall)

    ax.set_xlabel("best recall @ 1")
    ax.set_ylabel("training time (seconds)")
    if not best_recall:
        ax.text(0.5, 0.5, "No search experiments", ha="center", va="center")
        return

    marker = itertools.cycle(
        ("o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D")
    )
    training_times = []
    for index_name, recall in sorted(best_recall.items(), key=lambda item: item[1]):
        meta = results["indices"].get(index_name, {})
        training_time = meta.get("training_time")
        if training_time is None:
            continue
        ax.plot(
            [recall],
            [training_time],
            marker=next(marker),
            label=index_name,
            linewidth=0,
        )
        training_times.append(training_time)

    if not training_times:
        ax.text(0.5, 0.5, "No training-time metadata", ha="center", va="center")
        return
    if training_times and all(value > 0 for value in training_times):
        ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")


def save_plot(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


def write_default_plots(results, plot_dir):
    plt = load_plot_dependencies()
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (
            "knn_pareto_time.png",
            lambda: filter_results(
                results,
                evaluation="knn",
                accuracy_metric="knn_intersection",
                pareto_mode=ParetoMode.GLOBAL,
                pareto_metric=ParetoMetric.TIME,
            ),
            lambda ax, experiments: plot_metric(
                ax,
                experiments,
                accuracy_title="recall @ 1",
                cost_title="search time (seconds)",
            ),
        ),
        (
            "knn_pareto_space.png",
            lambda: filter_results(
                results,
                evaluation="knn",
                accuracy_metric="knn_intersection",
                pareto_mode=ParetoMode.GLOBAL,
                pareto_metric=ParetoMetric.SPACE,
            ),
            lambda ax, experiments: plot_metric(
                ax,
                experiments,
                accuracy_title="recall @ 1",
                cost_title="bytes per code",
                plot_space=True,
            ),
        ),
        (
            "rec_pareto_space.png",
            lambda: filter_results(
                results,
                evaluation="rec",
                accuracy_metric="sym_recall",
                time_metric=lambda experiment: experiment["encode_time"],
                pareto_mode=ParetoMode.GLOBAL,
                pareto_metric=ParetoMetric.SPACE,
            ),
            lambda ax, experiments: plot_metric(
                ax,
                experiments,
                accuracy_title="sym recall",
                cost_title="bytes per code",
                plot_space=True,
            ),
        ),
    ]

    written = []
    for file_name, load_experiments, render in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        render(ax, load_experiments())
        output_path = plot_dir / file_name
        save_plot(fig, output_path)
        plt.close(fig)
        written.append(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_codec_mse_vs_encode_time(ax, results)
    output_path = plot_dir / "codec_mse_vs_encode_time.png"
    save_plot(fig, output_path)
    plt.close(fig)
    written.append(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_training_time_vs_recall(ax, results)
    output_path = plot_dir / "training_time_vs_recall.png"
    save_plot(fig, output_path)
    plt.close(fig)
    written.append(output_path)
    return written


def download_hf_dataset(dataset_name):
    if dataset_name == DatasetSIFT1M.reference_name:
        DatasetSIFT1M.download()
    elif dataset_name == DatasetGlove100_6B.reference_name:
        DatasetGlove100_6B.download()
    elif dataset_name == DatasetGlove200_6B.reference_name:
        DatasetGlove200_6B.download()
    elif dataset_name == "dbpedia-1536-1M":
        DatasetDBpedia1536_1M.download()
    elif dataset_name == "dbpedia-3072-1M":
        DatasetDBpedia3072_1M.download()


def make_dataset(dataset_name, num_samples=None, num_queries=None, metric=None):
    self_retrieval_nq = (
        num_queries if num_queries is not None else DEFAULT_SELF_RETRIEVAL_NQ
    )
    if dataset_name == DatasetGlove100_6B.reference_name:
        return DatasetGlove100_6B(
            num_samples=num_samples,
            num_queries=self_retrieval_nq,
            metric=metric or "IP",
        )
    if dataset_name == DatasetGlove200_6B.reference_name:
        return DatasetGlove200_6B(
            num_samples=num_samples,
            num_queries=self_retrieval_nq,
            metric=metric or "IP",
        )
    if dataset_name == "dbpedia-1536-1M":
        return DatasetDBpedia1536_1M(
            num_samples=num_samples,
            num_queries=self_retrieval_nq,
            metric=metric or "IP",
        )
    if dataset_name == "dbpedia-3072-1M":
        return DatasetDBpedia3072_1M(
            num_samples=num_samples,
            num_queries=self_retrieval_nq,
            metric=metric or "IP",
        )
    if dataset_name == DatasetSIFT1M.reference_name:
        return DatasetSIFT1M()

    raise ValueError("invalid database name")


def main():
    global RESULT_REPORTER

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-mode",
        choices=("codec", "search", "both"),
        default="codec",
    )
    parser.add_argument(
        "--metric",
        type=normalize_metric_name,
        choices=("l2", "ip"),
        default=None,
        help=(
            "Search metric. For fixed-ground-truth datasets this is validated "
            "against the dataset. For self-retrieval datasets loaded from "
            "Hugging Face it overrides the default metric."
        ),
    )
    parser.add_argument("--query-count", type=int, default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of corpus vectors to benchmark",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download supported Hugging Face datasets into the default cache",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set the Faiss OpenMP thread count with faiss.omp_set_num_threads",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional debug artifact: write one JSON object per emitted result row."
        ),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Write the bench_fw-compatible result.json artifact here.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for static plot outputs.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Read --result-json and regenerate plots without rerunning benchmarks.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip static plot generation after benchmarking.",
    )
    parser.add_argument(
        "--seed",
        "--sample-seed",
        dest="seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--recall-at",
        type=str,
        default=",".join(str(k) for k in DEFAULT_RECALL_AT),
    )
    parser.add_argument(
        "--search-k",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--nprobe-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_NPROBE_VALUES),
    )
    parser.add_argument("--nlist", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--aq-search-type",
        default="norm_qint8",
        help=(
            "Search type for AQ-based search indexes: "
            + ", ".join(sorted(AQ_SEARCH_TYPE_CHOICES))
        ),
    )
    parser.add_argument("--qb", type=int, default=8)
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--rabitq-rrot-seed", type=int, default=123)
    parser.add_argument("--no-rabitq-rrot", action="store_true")
    parser.add_argument(
        "index_specs",
        nargs="*",
        help=(
            "Optional dataset name plus explicit index specs like "
            "RQ42x12, PRQ32x4x8, TQMSE8, or RaBitQ8"
        ),
    )
    args = parser.parse_args()
    index_specs = list(args.index_specs)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_json_path = args.result_json or default_output_path(run_id, "json")

    dataset_name = DatasetSIFT1M.reference_name
    if len(index_specs) > 0 and index_specs[0] in DATASET_NAMES:
        dataset_name = index_specs[0]
        del index_specs[0]
    args.dataset_name = dataset_name

    if args.plot_only:
        if args.result_json is None:
            raise RuntimeError("--plot-only requires --result-json")
        results = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
        if args.no_plots:
            return
        plot_dir = args.plot_dir or default_plot_dir(args.result_json)
        emit_progress(
            [
                ("progress_phase", "plot_start"),
                ("result_json_path", str(args.result_json)),
                ("plot_dir", str(plot_dir)),
            ]
        )
        written_plots = write_default_plots(results, plot_dir)
        emit_progress(
            [
                ("progress_phase", "plot_done"),
                ("result_json_path", str(args.result_json)),
                ("plot_dir", str(plot_dir)),
                ("plot_count", len(written_plots)),
            ]
        )
        return

    configured_threads = configure_omp_threads(args.threads)

    validate_requested_metric(dataset_name, args.metric)

    index_specs, ignored_index_specs = parse_index_specs(index_specs)
    if ignored_index_specs:
        print(
            "ignored_index_specs=" + ",".join(sorted(ignored_index_specs)),
            file=sys.stderr,
        )

    if args.download and dataset_name in DOWNLOADABLE_DATASET_NAMES:
        emit_progress(
            [
                ("progress_phase", "download_start"),
                ("dataset", normalize_name(dataset_name)),
            ]
        )
        download_hf_dataset(dataset_name)
        emit_progress(
            [
                ("progress_phase", "download_done"),
                ("dataset", normalize_name(dataset_name)),
            ]
        )

    if not index_specs:
        if ignored_index_specs or args.download:
            return
        raise RuntimeError("no recognized index specs were provided")

    emit_progress(
        [
            ("progress_phase", "dataset_load_start"),
            ("dataset", normalize_name(dataset_name)),
        ]
    )
    ds = make_dataset(
        dataset_name,
        num_samples=args.num_samples,
        num_queries=args.query_count,
        metric=args.metric,
    )
    emit_progress(
        [
            ("progress_phase", "dataset_load_done"),
            ("dataset", normalize_name(dataset_name)),
            ("dimension", ds.d),
            ("dataset_database_size", ds.nb),
            ("dataset_query_count", ds.nq),
        ]
    )

    recall_at = parse_int_list(args.recall_at, "--recall-at")
    search_k_values = (
        parse_int_list(args.search_k, "--search-k")
        if args.search_k
        else (max(recall_at),)
    )
    args.nprobe_values = parse_int_list(args.nprobe_values, "--nprobe-values")

    if args.trials <= 0:
        raise RuntimeError("--trials must be positive")
    if args.qb < 0:
        raise RuntimeError("--qb must be non-negative")
    if args.nlist <= 0:
        raise RuntimeError("--nlist must be positive")
    if args.rabitq_rrot_seed < 0:
        raise RuntimeError("--rabitq-rrot-seed must be non-negative")
    for index_spec in index_specs:
        validate_index_spec(ds.d, index_spec, args.benchmark_mode)

    needs_aq_search_type = any(
        index_spec.family in {"rq", "rq_lut", "lsq", "prq", "plsq"}
        for index_spec in index_specs
    ) and args.benchmark_mode in {"search", "both"}
    args.aq_search_type_name = normalize_name(args.aq_search_type)
    if needs_aq_search_type:
        args.aq_search_type = resolve_aq_search_type(args.aq_search_type)

    metric_type = get_metric_type(ds)
    rng = np.random.RandomState(args.seed)
    emit_progress(
        [
            ("progress_phase", "evaluation_data_start"),
            ("dataset", normalize_name(dataset_name)),
        ]
    )
    xb, xq, gt, query_mode = build_eval_data(
        ds,
        query_count=args.query_count,
        num_samples=args.num_samples,
        max_gt_k=max(recall_at),
        metric_type=metric_type,
        rng=rng,
    )
    emit_progress(
        [
            ("progress_phase", "evaluation_data_done"),
            ("database_size", xb.shape[0]),
            ("query_count", xq.shape[0]),
            ("query_mode", normalize_name(query_mode)),
        ]
    )
    max_nbits = max(
        index_spec.nbits
        for index_spec in index_specs
        if index_spec.nbits is not None
    )
    maxtrain = max(100 << max_nbits, 10**5)
    emit_progress(
        [
            ("progress_phase", "training_data_start"),
            ("maxtrain_requested", maxtrain),
        ]
    )
    xt, training_source = get_training_vectors(
        ds,
        xb,
        maxtrain=maxtrain,
    )
    emit_progress(
        [
            ("progress_phase", "training_data_done"),
            ("train_size", xt.shape[0]),
            ("training_source", normalize_name(training_source)),
        ]
    )

    nb, d = xb.shape
    nq = xq.shape[0]
    nt = xt.shape[0]
    ivf_nlist = max(1, min(args.nlist, nb, nt))

    if max(search_k_values) > nb:
        raise RuntimeError(
            f"max search_k={max(search_k_values)} exceeds database size {nb}"
        )

    base_fields = [
        ("dataset", normalize_name(dataset_name)),
        ("metric", normalize_name(ds.metric)),
        ("dimension", d),
        ("database_size", nb),
        ("query_count", nq),
        ("train_size", nt),
        ("maxtrain_requested", maxtrain),
        ("query_mode", normalize_name(query_mode)),
        ("training_source", normalize_name(training_source)),
        ("sample_seed", args.seed),
    ]
    if configured_threads is not None:
        base_fields.append(("omp_threads", configured_threads))
    if args.num_samples is not None:
        base_fields.append(("num_samples", args.num_samples))
    base_fields.append(("recall_at", "_".join(str(k) for k in recall_at)))

    search_jobs = []
    if args.benchmark_mode in {"search", "both"}:
        search_jobs = build_search_jobs(
            d,
            metric_type,
            index_specs,
            args,
            ivf_nlist,
        )

    codec_job_count = 0
    codec_result_count = 0
    if args.benchmark_mode in {"codec", "both"}:
        codec_job_count, codec_result_count = count_codec_jobs_and_results(
            index_specs
        )

    search_job_count = 0
    search_result_count = 0
    if search_jobs:
        search_job_count = len(search_jobs)
        search_result_count = sum(
            count_search_results(
                search_k_values,
                recall_at,
                len(make_search_points(index_kind, index_spec.family, args)),
            )
            for index_spec, index_name, index_kind, factory, spec_fields in search_jobs
        )

    RESULT_REPORTER = ResultReporter(
        result_json_path=result_json_path,
        output_jsonl_path=args.output_jsonl,
        total_jobs=codec_job_count + search_job_count,
        total_results=codec_result_count + search_result_count,
    )

    results = None
    try:
        if args.benchmark_mode in {"codec", "both"}:
            for index_spec in index_specs:
                if index_spec.family == "lsq-gpu":
                    lsq = faiss.LocalSearchQuantizer(
                        d, index_spec.M, index_spec.nbits
                    )
                    ngpus = faiss.get_num_gpus()
                    lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
                    eval_quantizer(
                        lsq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        extra_fields=[("gpu_count", ngpus)],
                    )
                elif index_spec.family == "pq":
                    pq = faiss.ProductQuantizer(d, index_spec.M, index_spec.nbits)
                    eval_quantizer(
                        pq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                    )
                elif index_spec.family == "opq":
                    d2 = ((d + index_spec.M - 1) // index_spec.M) * index_spec.M
                    t0_ns = now_ns()
                    opq = faiss.OPQMatrix(d, index_spec.M, d2)
                    opq.train(xt)
                    xq2 = opq.apply(xq)
                    xb2 = opq.apply(xb)
                    xt2 = opq.apply(xt)
                    opq_train_time_ms = elapsed_ms(t0_ns)
                    pq = faiss.ProductQuantizer(d2, index_spec.M, index_spec.nbits)
                    eval_quantizer(
                        pq,
                        index_spec,
                        xq2,
                        xb2,
                        gt,
                        xt2,
                        metric_type,
                        recall_at,
                        base_fields,
                        train_time_offset_ms=opq_train_time_ms,
                        extra_fields=[("projected_dimension", d2)],
                    )
                elif index_spec.family == "prq":
                    prq = faiss.ProductResidualQuantizer(
                        d,
                        index_spec.nsplits,
                        index_spec.Msub,
                        index_spec.nbits,
                    )
                    variants = [("max_beam_size", i) for i in (1, 2, 4, 8, 16, 32)]
                    eval_quantizer(
                        prq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        variants=variants,
                    )
                elif index_spec.family == "plsq":
                    plsq = faiss.ProductLocalSearchQuantizer(
                        d,
                        index_spec.nsplits,
                        index_spec.Msub,
                        index_spec.nbits,
                    )
                    variants = [("encode_ils_iters", i) for i in (2, 3, 4, 8, 16)]
                    eval_quantizer(
                        plsq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        variants=variants,
                    )
                elif index_spec.family == "rq":
                    rq = faiss.ResidualQuantizer(d, index_spec.M, index_spec.nbits)
                    rq.max_beam_size = 30
                    variants = [("max_beam_size", i) for i in (1, 2, 4, 8, 16, 32)]
                    eval_quantizer(
                        rq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        variants=variants,
                    )
                elif index_spec.family == "rq_lut":
                    rq = faiss.ResidualQuantizer(d, index_spec.M, index_spec.nbits)
                    rq.max_beam_size = 30
                    rq.use_beam_LUT = 1
                    variants = [("max_beam_size", i) for i in (1, 2, 4, 8, 16, 32, 64)]
                    eval_quantizer(
                        rq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        variants=variants,
                        extra_fields=[("use_beam_lut", 1)],
                    )
                elif index_spec.family == "lsq":
                    lsq = faiss.LocalSearchQuantizer(
                        d, index_spec.M, index_spec.nbits
                    )
                    variants = [("encode_ils_iters", i) for i in (2, 3, 4, 8, 16)]
                    eval_quantizer(
                        lsq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        variants=variants,
                    )
                elif index_spec.family == "tqmse":
                    store_norm = dataset_name not in {
                        DatasetGlove100_6B.reference_name,
                        DatasetGlove200_6B.reference_name,
                    }
                    if not hasattr(faiss, "IndexTurboQuantMSE"):
                        raise RuntimeError(
                            "TurboQuantMSE is not available in this faiss Python build. "
                            "Rebuild the Python bindings so IndexTurboQuantMSE is exported."
                        )
                    tqmse = faiss.IndexTurboQuantMSE(
                        d,
                        index_spec.nbits,
                        metric_type,
                        args.seed,
                        store_norm,
                    )
                    eval_quantizer(
                        tqmse,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        extra_fields=[("store_norm", int(store_norm))],
                    )
                elif index_spec.family == "tqprod":
                    store_norm = dataset_name not in {
                        DatasetGlove100_6B.reference_name,
                        DatasetGlove200_6B.reference_name,
                    }
                    if not hasattr(faiss, "IndexTurboQuantProd"):
                        raise RuntimeError(
                            "TurboQuantProd is not available in this faiss Python build. "
                            "Rebuild the Python bindings so IndexTurboQuantProd is exported."
                        )
                    tqprod = faiss.IndexTurboQuantProd(
                        d,
                        index_spec.nbits,
                        metric_type,
                        args.seed,
                        store_norm,
                    )
                    eval_quantizer(
                        tqprod,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        extra_fields=[("store_norm", int(store_norm))],
                    )
                elif index_spec.family == "rabitq":
                    if not hasattr(faiss, "IndexRaBitQ"):
                        raise RuntimeError(
                            "RaBitQ is not available in this faiss Python build. "
                            "Rebuild the Python bindings so RaBitQ symbols are exported."
                        )
                    rbq = faiss.IndexRaBitQ(d, metric_type, index_spec.nbits)
                    eval_quantizer(
                        rbq,
                        index_spec,
                        xq,
                        xb,
                        gt,
                        xt,
                        metric_type,
                        recall_at,
                        base_fields,
                        extra_fields=[("qb", args.qb), ("centered", int(args.centered))],
                    )

        if args.benchmark_mode in {"search", "both"}:
            for index_spec, index_name, index_kind, factory, spec_fields in search_jobs:
                index = factory()
                configure_search_index(index, index_spec.family, args)
                spec_fields = train_and_add_index(index, xt, xb) + spec_fields
                points = make_search_points(index_kind, index_spec.family, args)
                benchmark_search_index(
                    index,
                    index_spec,
                    index_name,
                    index_kind,
                    xq,
                    gt,
                    recall_at,
                    search_k_values,
                    args.trials,
                    base_fields,
                    spec_fields,
                    points,
                )
    finally:
        if RESULT_REPORTER is not None:
            results = RESULT_REPORTER.results
            RESULT_REPORTER.close()
            RESULT_REPORTER = None

    if args.no_plots:
        return

    plot_dir = args.plot_dir or default_plot_dir(result_json_path)
    emit_progress(
        [
            ("progress_phase", "plot_start"),
            ("result_json_path", str(result_json_path)),
            ("plot_dir", str(plot_dir)),
        ]
    )
    written_plots = write_default_plots(results, plot_dir)
    emit_progress(
        [
            ("progress_phase", "plot_done"),
            ("result_json_path", str(result_json_path)),
            ("plot_dir", str(plot_dir)),
            ("plot_count", len(written_plots)),
        ]
    )

if __name__ == "__main__":
    main()
