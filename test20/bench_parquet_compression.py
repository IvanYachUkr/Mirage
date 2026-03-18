"""Quick benchmark: Feather (lz4/zstd/none) vs Parquet (zstd-1) write+read speed."""
import time, os, gc, shutil, tempfile
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pyarrow.feather as feather

N_ROWS = 2_000_000
BATCH_SIZE = 10_000
N_BATCHES = N_ROWS // BATCH_SIZE

SCHEMA = pa.schema([
    ("title_id", pa.int32()),
    ("reviewer_type", pa.string()),
    ("source", pa.string()),
    ("rating_10", pa.float32()),
    ("sentiment", pa.float32()),
    ("review_date", pa.string()),
    ("review_text", pa.string()),
])

def generate_batch(rng, batch_size=BATCH_SIZE):
    sources = ["Variety", "NYT", "IndieWire", "Empire", "user", "The Guardian",
               "Hollywood Reporter", "RogerEbert.com", "Slant Magazine", "Total Film"]
    adjectives = ["sharp", "moving", "flat", "bold", "riveting", "hollow", "gripping",
                  "masterful", "pedestrian", "captivating", "bloated", "luminous"]
    title_ids = rng.randint(1, 200_001, size=batch_size).tolist()
    reviewer_types = rng.choice(["critic", "audience"], size=batch_size, p=[0.15, 0.85]).tolist()
    source_list = rng.choice(sources, size=batch_size).tolist()
    ratings = np.clip(rng.normal(6.5, 1.5, size=batch_size), 0, 10).astype(np.float32).tolist()
    sentiments = np.clip((np.array(ratings) - 5.0) / 2.5, -1, 1).astype(np.float32).tolist()
    dates = [f"20{rng.randint(15, 26):02d}-{rng.randint(1, 13):02d}-{rng.randint(1, 29):02d}"
             for _ in range(batch_size)]
    texts = []
    for i in range(batch_size):
        adj = rng.choice(adjectives)
        if rng.random() < 0.3:
            texts.append(f"This film is {adj}. A {adj} piece of filmmaking. Score: {ratings[i]:.1f}/10. "
                         f"The performances are {rng.choice(adjectives)} throughout.")
        else:
            texts.append(f"{adj} -- {ratings[i]:.1f}/10.")
    return pa.RecordBatch.from_pydict({
        "title_id": title_ids, "reviewer_type": reviewer_types, "source": source_list,
        "rating_10": ratings, "sentiment": sentiments, "review_date": dates, "review_text": texts,
    }, schema=SCHEMA)


def bench_feather_streaming(batches, path, compression):
    """Stream batches via RecordBatchFileWriter (Feather v2 = Arrow IPC)."""
    t0 = time.perf_counter()
    opts = ipc.IpcWriteOptions(compression=compression) if compression else ipc.IpcWriteOptions()
    with pa.OSFile(path, 'wb') as f:
        writer = ipc.new_file(f, SCHEMA, options=opts)
        for batch in batches:
            writer.write_batch(batch)
        writer.close()
    write_s = time.perf_counter() - t0
    file_bytes = os.path.getsize(path)
    # Read
    t0 = time.perf_counter()
    table = feather.read_table(path)
    read_s = time.perf_counter() - t0
    assert len(table) == N_ROWS
    del table; gc.collect()
    return write_s, read_s, file_bytes


def bench_parquet_streaming(batches, path, compression, level=None):
    t0 = time.perf_counter()
    kw = {"compression_level": level} if level is not None else {}
    writer = pq.ParquetWriter(path, SCHEMA, compression=compression, **kw)
    for batch in batches:
        writer.write_batch(batch)
    writer.close()
    write_s = time.perf_counter() - t0
    file_bytes = os.path.getsize(path)
    t0 = time.perf_counter()
    table = pq.read_table(path)
    read_s = time.perf_counter() - t0
    assert len(table) == N_ROWS
    del table; gc.collect()
    return write_s, read_s, file_bytes


def main():
    rng = np.random.RandomState(42)
    print(f"Generating {N_BATCHES} batches x {BATCH_SIZE} = {N_ROWS:,} rows...")
    batches = [generate_batch(rng) for _ in range(N_BATCHES)]
    raw_mb = sum(b.nbytes for b in batches) / 1e6
    print(f"  Raw Arrow: {raw_mb:.1f} MB\n")

    tmpdir = tempfile.mkdtemp(prefix="feather_bench_")
    results = []

    configs = [
        ("Feather none",     "feather", None, None),
        ("Feather lz4",      "feather", "lz4", None),
        ("Feather zstd",     "feather", "zstd", None),
        ("Parquet zstd-1",   "parquet", "zstd", 1),
        ("Parquet snappy",   "parquet", "snappy", None),
    ]

    for label, fmt, comp, level in configs:
        ext = ".arrow" if fmt == "feather" else ".parquet"
        path = os.path.join(tmpdir, f"bench_{label.replace(' ', '_')}{ext}")
        try:
            if fmt == "feather":
                ws, rs, fs = bench_feather_streaming(batches, path, comp)
            else:
                ws, rs, fs = bench_parquet_streaming(batches, path, comp, level)
            results.append((label, ws, rs, fs))
            print(f"  {label:20s}  write={ws:.3f}s  read={rs:.3f}s  size={fs/1e6:.1f}MB")
        except Exception as e:
            print(f"  {label:20s}  FAILED: {e}")
        try:
            os.remove(path)
        except OSError:
            pass

    print(f"\n{'='*85}")
    print(f"{'Format':20s} {'Write(s)':>9s} {'Read(s)':>9s} {'Size(MB)':>9s} {'Write MB/s':>11s} {'Read MB/s':>11s}")
    print(f"{'='*85}")
    for label, ws, rs, fs in sorted(results, key=lambda x: x[1]):
        print(f"{label:20s} {ws:9.3f} {rs:9.3f} {fs/1e6:9.1f} {raw_mb/ws:11.1f} {raw_mb/rs:11.1f}")

    shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
