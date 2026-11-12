# PS-HR prototype - build, test, and benchmark targets
PY  ?= python3
CC  ?= gcc
DATA ?= data/banking77
EMBED ?=
OUT ?= results/real
BUILD ?= build
ARCH := $(shell uname -m)
ifneq (,$(filter aarch64 arm64,$(ARCH)))
SCALAR_FLAGS = -O2 -fno-tree-vectorize -fno-tree-slp-vectorize
SIMD_FLAGS   = -O3 -mcpu=native
else
SCALAR_FLAGS = -O2 -mpopcnt -fno-tree-vectorize -fno-tree-slp-vectorize
SIMD_FLAGS   = -O3 -march=native
endif
SEEDS ?= 3

.PHONY: all test experiments experiments-knowledge experiments-semantic experiments-compositional experiments-qa experiments-embedding experiments-real build bench sweep zip clean help

help:
	@echo "make test              unit tests (seconds)"
	@echo "make experiments       Python experiment suite on synthetic data -> results/ (several minutes)"
	@echo "make experiments-real  same suite on DATA=<prefix> (expects <prefix>_train.tsv / _test.tsv)"
	@echo "make experiments-knowledge  sharded holographic knowledge memory: multi-hop, corruption, cascade cleanup"
	@echo "make experiments-semantic   random-indexing / IDF negative-result experiments (needs local Python docs corpus)"
	@echo "make experiments-compositional  H5: unseen action+object combinations"
	@echo "make experiments-qa         end-to-end text question answering"
	@echo "make bench             C dense-vs-cascade benchmark (scalar + SIMD builds, cold cache)"
	@echo "make sweep             C near-tie / prefix / adaptive-margin sweep"
	@echo "make zip               package the project as pshr.zip one level up"
	@echo "make clean"

all: test bench

test:
	$(PY) -m unittest discover -s tests -v

experiments:
	cd experiments && $(PY) run_experiments.py --seeds $(SEEDS) --out ../results

experiments-knowledge:
	cd experiments && $(PY) run_knowledge.py

experiments-semantic:
	mkdir -p work && cd work && $(PY) ../scripts/build_corpus.py && $(PY) ../experiments/run_semantic.py

experiments-compositional:
	cd experiments && $(PY) run_compositional.py

experiments-qa:
	cd experiments && $(PY) run_qa.py

experiments-embedding:
	@echo "Stand-in check (needs ri.npz from experiments-semantic):  cd work && python3 ../experiments/run_embedding.py --model ri"
	@echo "Real model (needs network + sentence-transformers): cd experiments && python3 run_embedding.py --model sentence-transformers/all-MiniLM-L6-v2"

experiments-real:
	cd experiments && $(PY) run_experiments.py --data ../$(DATA)_train.tsv ../$(DATA)_test.tsv --seeds $(SEEDS) $(if $(EMBED),--embed-model $(EMBED),) --out ../$(OUT)

build:
	mkdir -p $(BUILD)
	$(CC) $(SCALAR_FLAGS) -o $(BUILD)/bench_scalar bench/pshr_bench.c
	$(CC) $(SIMD_FLAGS) -o $(BUILD)/bench_simd bench/pshr_bench.c
	$(CC) $(SCALAR_FLAGS) -o $(BUILD)/sweep_scalar bench/pshr_sweep.c -lm
	$(CC) $(SIMD_FLAGS) -o $(BUILD)/sweep_simd bench/pshr_sweep.c -lm

# On ARM replace -mpopcnt / -march=native as appropriate, e.g.:  make build CC=gcc  (edit flags) or
# gcc -O2 -mcpu=native -fno-tree-vectorize -o $(BUILD)/bench_scalar bench/pshr_bench.c

bench: build
	@echo "== scalar, cold cache =="; $(BUILD)/bench_scalar 100000 1
	@echo "== SIMD, cold cache ==";   $(BUILD)/bench_simd 100000 1

sweep: build
	@echo "== near-tie sweep (scalar, cold): args N cluster k_intra k_query prefix_words cold =="
	$(BUILD)/sweep_scalar 100000 20000 5 2 16 1
	$(BUILD)/sweep_scalar 100000 20000 6 2 16 1

zip: clean
	cd .. && rm -f pshr.zip && zip -qr pshr.zip $(notdir $(CURDIR)) -x '*/__pycache__/*' '*/$(BUILD)/*' '*/data/*'
	@echo "wrote ../pshr.zip"

clean:
	rm -rf $(BUILD) $$(find . -name __pycache__)
