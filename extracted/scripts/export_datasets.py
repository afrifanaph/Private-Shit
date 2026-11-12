"""
Export Banking77 and CLINC150 to the TSV format run_experiments.py expects ('label<TAB>text').
Needs network access and:  pip install datasets
NOTE: written for the Hugging Face dataset schemas as I know them; it could not be tested in the
sandbox where this project was built (no network). If a field name differs, adjust the two
`text_key`/`label_key` values below.

Usage:  python scripts/export_datasets.py data/
Then:   make experiments-real DATA=data/banking77
"""
import os, sys
from datasets import load_dataset

out = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(out, exist_ok=True)


def dump(ds, path, text_key, label_key, names):
    with open(path, "w", encoding="utf8") as f:
        for r in ds:
            lab = names[r[label_key]] if names else r[label_key]
            f.write(f"{lab}\t{r[text_key].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    print("wrote", path, len(ds))


b = load_dataset("PolyAI/banking77")
names = b["train"].features["label"].names
dump(b["train"], f"{out}/banking77_train.tsv", "text", "label", names)
dump(b["test"], f"{out}/banking77_test.tsv", "text", "label", names)

c = load_dataset("clinc_oos", "plus")
names = c["train"].features["intent"].names
dump(c["train"], f"{out}/clinc150_train.tsv", "text", "intent", names)
dump(c["test"], f"{out}/clinc150_test.tsv", "text", "intent", names)
