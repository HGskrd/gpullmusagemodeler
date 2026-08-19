"""Token buckets and request-distribution presets."""

from dataclasses import dataclass


@dataclass
class Bucket:
    length: int
    label: str
    color: str


INPUT_BUCKETS = [
    Bucket(256, "256", "#10825c"),
    Bucket(1024, "1k", "#1D9E75"),
    Bucket(4096, "4k", "#3266ad"),
    Bucket(16384, "16k", "#7F77DD"),
    Bucket(32768, "32k", "#BA7517"),
    Bucket(65536, "64k", "#D85A30"),
    Bucket(131072, "128k", "#A32D2D"),
]

OUTPUT_BUCKETS = [
    Bucket(32, "32", "#10825c"),
    Bucket(128, "128", "#1D9E75"),
    Bucket(512, "512", "#3266ad"),
    Bucket(2048, "2k", "#7F77DD"),
    Bucket(4096, "4k", "#BA7517"),
    Bucket(8192, "8k", "#D85A30"),
]

EMBEDDING_DOC_BUCKETS = [
    Bucket(32, "32", "#10825c"),
    Bucket(128, "128", "#1D9E75"),
    Bucket(256, "256", "#3266ad"),
    Bucket(1024, "1k", "#7F77DD"),
    Bucket(2048, "2k", "#BA7517"),
    Bucket(8192, "8k", "#D85A30"),
    Bucket(32768, "32k", "#A32D2D"),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

DIST_PRESETS = {
    "Chat": {"in": [10, 30, 35, 15, 7, 2, 1], "out": [15, 30, 35, 15, 4, 1]},
    "RAG": {"in": [5, 15, 25, 25, 18, 8, 4], "out": [10, 25, 40, 18, 5, 2]},
    "Long doc": {"in": [2, 5, 10, 15, 25, 25, 18], "out": [5, 15, 30, 30, 15, 5]},
    "Code": {"in": [8, 25, 30, 22, 10, 4, 1], "out": [10, 20, 35, 25, 8, 2]},
    "Classify": {"in": [5, 20, 40, 25, 8, 2, 0], "out": [80, 15, 4, 1, 0, 0]},
}

EMBEDDING_DOC_PRESETS = {
    "Query": [90, 10, 0, 0, 0, 0, 0],
    "Passage": [0, 10, 75, 15, 0, 0, 0],
    "Doc": [0, 5, 15, 40, 30, 10, 0],
    "Long doc": [0, 0, 2, 8, 15, 55, 20],
}

TASK_PRESETS = {
    "Classify": {"i": 2048, "o": 32},
    "Extract": {"i": 4096, "o": 256},
    "Summarize": {"i": 8192, "o": 512},
    "Rephrase": {"i": 2048, "o": 2048},
    "Synth gen": {"i": 512, "o": 4096},
    "Score": {"i": 4096, "o": 8},
}
