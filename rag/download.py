"""Download the five NFCorpus inputs from fixed BEIR dataset revisions."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


CORPUS_URL = (
    "https://huggingface.co/datasets/BeIR/nfcorpus/resolve/"
    "e28763e68d85db0fa71652ba3c4afabf8c5b3bb7"
)
QRELS_URL = (
    "https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/"
    "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65"
)
FILES = {
    "corpus.jsonl.gz": (CORPUS_URL, "ae4d6ab2ab88d880e7e180776c956a09c19c43a703457742cd850d2422035802"),
    "queries.jsonl.gz": (CORPUS_URL, "baba3504a2736b0e32265962f04acc4fecfaf9649e0ea08e49ff441246f75364"),
    "dev.tsv": (QRELS_URL, "b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8"),
    "train.tsv": (QRELS_URL, "6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a"),
    "test.tsv": (QRELS_URL, "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a"),
}


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (base_url, expected_hash) in FILES.items():
        destination = directory / name
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_hash:
            print(f"Verified {name}")
            continue
        with urlopen(f"{base_url}/{name}", timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Checksum mismatch for {name}")
        with NamedTemporaryFile(dir=directory, delete=False) as temp:
            temp.write(data)
        Path(temp.name).replace(destination)
        print(f"Downloaded and verified {name}")
