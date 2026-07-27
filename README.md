# Experimental analysis of a tensor network-based search method

This repository contains the code used for the experiments in the thesis "Experimental analysis of a tensor network-based search method". The experiments evaluate whether a tensor network method, namely the Quantum-inspired Grover Algorithm (QiGA), is an effective alternative to Grover's algorithm. The code implements the MD5 preimage attack as a benchmark, chosen as a worst-case scenario for QiGA since MD5 produces a highly entangled state. The experiments compare the quantum, QiGA, and brute-force implementations of the attack. Additionally, the experiments measure how the MPS bond dimension affects execution time and accuracy.

The `results` directory contains the output of the experiments. The final results were inconclusive, as executing MD5 on the superposition state using a tensor network took much longer than attacking MD5 using a quantum computer. However, after this step, checking different hashes was quicker. Furthermore, the results show that, for the highly entangled state produced by MD5, it is challenging to limit the bond dimension without significantly worsening the accuracy of the results.

- `benchmark_mps.py` : QiGA implementation of MD5 preimage attack. 
- `run_mps.slurm` : Slurm launcher for `benchmark_mps.py`.
- `benchmark_bruteforce.py` : Classical brute-force MD5 preimage attack. `tiny_md5_bits` is the single source of truth for the classical hash and is used for preimage verification.
- `benchmark_qiskit.py` : Full Grover search, run on real IBM Quantum hardware (or a simulator). Requires IBM Quantum credentials configured for `QiskitRuntimeService()`.

Usage examples:

```
python benchmark_mps.py --n 1 --w 16 --mps-bonds 8 16 32 64 --include-uncompressed --n-hashes 20
python benchmark_mps.py --n 1 --w 16 --mps-bonds 8 16 --query-mode sample --n-samples 5000 --top-k 1000
python benchmark_bruteforce.py --n 1 --w 16 --mode both
python benchmark_qiskit.py --n 2 --w 32 --shots 3 --trials 10
```


