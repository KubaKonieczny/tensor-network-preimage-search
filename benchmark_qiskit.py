import argparse
import json
import random
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from benchmark_bruteforce import tiny_md5_bits


MD5_K = [
    0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE,
    0xF57C0FAF, 0x4787C62A, 0xA8304613, 0xFD469501,
    0x698098D8, 0x8B44F7AF, 0xFFFF5BB1, 0x895CD7BE,
    0x6B901122, 0xFD987193, 0xA679438E, 0x49B40821,
    0xF61E2562, 0xC040B340, 0x265E5A51, 0xE9B6C7AA,
    0xD62F105D, 0x02441453, 0xD8A1E681, 0xE7D3FBC8,
    0x21E1CDE6, 0xC33707D6, 0xF4D50D87, 0x455A14ED,
    0xA9E3E905, 0xFCEFA3F8, 0x676F02D9, 0x8D2A4C8A,
    0xFFFA3942, 0x8771F681, 0x6D9D6122, 0xFDE5380C,
    0xA4BEEA44, 0x4BDECFA9, 0xF6BB4B60, 0xBEBFBC70,
    0x289B7EC6, 0xEAA127FA, 0xD4EF3085, 0x04881D05,
    0xD9D4D039, 0xE6DB99E5, 0x1FA27CF8, 0xC4AC5665,
    0xF4292244, 0x432AFF97, 0xAB9423A7, 0xFC93A039,
    0x655B59C3, 0x8F0CCC92, 0xFFEFF47D, 0x85845DD1,
    0x6FA87E4F, 0xFE2CE6E0, 0xA3014314, 0x4E0811A1,
    0xF7537E82, 0xBD3AF235, 0x2AD7D2BB, 0xEB86D391,
]


MD5_S = [7, 12, 17, 22] * 4 + [5, 9, 14, 20] * 4 + [4, 11, 16, 23] * 4 + [6, 10, 15, 21] * 4


def get_md5_params(n):
    mask = (1 << n) - 1
    k_list = [(k >> (32 - n)) & mask for k in MD5_K]
    s_list = [max(1, min(n - 1, round(s * n / 32))) if n > 1 else 1 for s in MD5_S]
    return {"a": 0x67452301 & mask,
            "b": 0xEFCDAB89 & mask,
            "c": 0x98BADCFE & mask,
            "d": 0x10325476 & mask,
            "k_list": k_list,
            "s_list": s_list,
            "iterations": 64}


def get_chunk_index(i, num_chunks):
    if i < 16:
        return i % num_chunks
    elif i < 32:
        return (1 + 5 * i) % num_chunks
    elif i < 48:
        return (5 + 3 * i) % num_chunks
    else:
        return (7 * i) % num_chunks


def MAJ(qc, a, b, cin):
    qc.cx(a, b)
    qc.cx(a, cin)
    qc.ccx(cin, b, a)


def UMA(qc, a, b, cin):
    qc.ccx(cin, b, a)
    qc.cx(a, cin)
    qc.cx(cin, b)


def add(qc, n, target, added, carry):
    MAJ(qc, added[0], target[0], carry)
    for i in range(n - 1):
        MAJ(qc, added[i + 1], target[i + 1], added[i])
    for i in reversed(range(n - 1)):
        UMA(qc, added[i + 1], target[i + 1], added[i])
    UMA(qc, added[0], target[0], carry)


def F(qc, b_reg, c_reg, d_reg):
    for i in range(len(b_reg)):
        qc.cx(d_reg[i], c_reg[i])
        qc.ccx(b_reg[i], c_reg[i], d_reg[i])


def F_uncompute(qc, b_reg, c_reg, d_reg):
    for i in range(len(b_reg)):
        qc.ccx(b_reg[i], c_reg[i], d_reg[i])
        qc.cx(d_reg[i], c_reg[i])


def G(qc, b_reg, c_reg, d_reg):
    for i in range(len(b_reg)):
        qc.cx(c_reg[i], b_reg[i])
        qc.ccx(b_reg[i], d_reg[i], c_reg[i])


def G_uncompute(qc, b_reg, c_reg, d_reg):
    for i in range(len(b_reg)):
        qc.ccx(b_reg[i], d_reg[i], c_reg[i])
        qc.cx(c_reg[i], b_reg[i])


def H(qc, b_reg, c_reg, d_reg):
    for i in range(len(b_reg)):
        qc.cx(b_reg[i], d_reg[i])
        qc.cx(c_reg[i], d_reg[i])


def I(qc, b_reg, c_reg, d_reg):
    for i in range(len(b_reg)):
        qc.x(c_reg[i])
        qc.x(b_reg[i])
        qc.ccx(b_reg[i], d_reg[i], c_reg[i])


def I_uncompute(qc, b_reg, c_reg, d_reg):
    for i in reversed(range(len(b_reg))):
        qc.ccx(b_reg[i], d_reg[i], c_reg[i])
        qc.x(b_reg[i])
        qc.x(c_reg[i])


def shift(qc, reg, shift_amount, n):
    shift_amount = shift_amount % n
    for _ in range(shift_amount):
        for i in range(len(reg) - 1, 0, -1):
            qc.swap(reg[i], reg[i - 1])


def add_k(qc, n, k, a_reg, temp_reg, carry):
    for i in range(n):
        if (k >> i) & 1:
            qc.x(temp_reg[i])
    add(qc, n, target=a_reg, added=temp_reg, carry=carry)
    for i in range(n):
        if (k >> i) & 1:
            qc.x(temp_reg[i])


def init_register(qc, n, val, reg):
    for i in range(n):
        if (val >> i) & 1:
            qc.x(reg[i])


def init_w(qc, w_reg):
    qc.h(w_reg)


def swap_regs(qc, a_reg, b_reg, c_reg, d_reg):
    for i in range(len(a_reg)):
        qc.swap(a_reg[i], b_reg[i])
        qc.swap(a_reg[i], c_reg[i])
        qc.swap(a_reg[i], d_reg[i])


def md5_iteration(qc, i, n, w_size, a_reg, b_reg, c_reg, d_reg, w_reg, temp, carry, k_list, s_list):
    num_chunks = w_size // n
    chunk_index = get_chunk_index(i, num_chunks)
    word_qubits = list(w_reg[chunk_index * n : chunk_index * n + n])

    add(qc, n, target=a_reg, added=word_qubits, carry=carry)

    if i < 16:
        F(qc, b_reg, c_reg, d_reg)
        add(qc, n, target=a_reg, added=d_reg, carry=carry)
        F_uncompute(qc, b_reg, c_reg, d_reg)
    elif i < 32:
        G(qc, b_reg, c_reg, d_reg)
        add(qc, n, target=a_reg, added=c_reg, carry=carry)
        G_uncompute(qc, b_reg, c_reg, d_reg)
    elif i < 48:
        H(qc, b_reg, c_reg, d_reg)
        add(qc, n, target=a_reg, added=d_reg, carry=carry)
        H(qc, b_reg, c_reg, d_reg)
    else:
        I(qc, b_reg, c_reg, d_reg)
        add(qc, n, target=a_reg, added=c_reg, carry=carry)
        I_uncompute(qc, b_reg, c_reg, d_reg)

    add_k(qc, n, k_list[i], a_reg, temp, carry)
    shift(qc, a_reg, s_list[i], n)
    add(qc, n, target=a_reg, added=b_reg, carry=carry)
    swap_regs(qc, a_reg, b_reg, c_reg, d_reg)


def add_iv(qc, n, a_reg, b_reg, c_reg, d_reg, iv_a_reg, iv_b_reg, iv_c_reg, iv_d_reg, carry):
    add(qc, n, target=a_reg, added=iv_a_reg, carry=carry)
    add(qc, n, target=b_reg, added=iv_b_reg, carry=carry)
    add(qc, n, target=c_reg, added=iv_c_reg, carry=carry)
    add(qc, n, target=d_reg, added=iv_d_reg, carry=carry)


def check_hash(qc, n, target_number, a_reg, b_reg, c_reg, d_reg, anc):
    outputs = [*a_reg, *b_reg, *c_reg, *d_reg]
    for i in range(n * 4):
        if ((target_number >> i) & 1) == 0:
            qc.x(outputs[i])
    qc.mcx(outputs, anc)
    for i in range(n * 4):
        if ((target_number >> i) & 1) == 0:
            qc.x(outputs[i])


def oracle(qc, target, w_size, n, a_reg, b_reg, c_reg, d_reg, w_reg, temp, carry, anc, iv_a_reg, iv_b_reg, iv_c_reg, iv_d_reg, iterations, k_list, s_list):
    for i in range(iterations):
        md5_iteration(qc, i, n, w_size, a_reg, b_reg, c_reg, d_reg, w_reg, temp, carry, k_list, s_list)
    add_iv(qc, n, a_reg, b_reg, c_reg, d_reg, iv_a_reg, iv_b_reg, iv_c_reg, iv_d_reg, carry)

    check_hash(qc, n, target, a_reg, b_reg, c_reg, d_reg, anc)

    tmp_qc = QuantumCircuit(a_reg, b_reg, c_reg, d_reg, w_reg, temp, carry, iv_a_reg, iv_b_reg, iv_c_reg, iv_d_reg)
    for i in range(iterations):
        md5_iteration(tmp_qc, i, n, w_size, a_reg, b_reg, c_reg, d_reg, w_reg, temp, carry, k_list, s_list)
    add_iv(tmp_qc, n, a_reg, b_reg, c_reg, d_reg, iv_a_reg, iv_b_reg, iv_c_reg, iv_d_reg, carry)

    qc.append(tmp_qc.inverse(), [*a_reg, *b_reg, *c_reg, *d_reg, *w_reg, *temp, *carry, *iv_a_reg, *iv_b_reg, *iv_c_reg, *iv_d_reg])


def diffusion(qc, w_reg):
    qc.h(w_reg)
    qc.x(w_reg)
    qc.h(w_reg[-1])
    qc.mcx(w_reg[:-1], w_reg[-1])
    qc.h(w_reg[-1])
    qc.x(w_reg)
    qc.h(w_reg)


def grover_search(n, w_size, a, b, c, d, target_hash, k_list, s_list, iterations):
    N = 2**w_size
    M = 2**w_size / 2 ** (4 * n)
    num_grover_iterations = int(np.pi / 4 * np.sqrt(N / M))

    w_reg = QuantumRegister(w_size, "w")
    a_reg = QuantumRegister(n, "a")
    b_reg = QuantumRegister(n, "b")
    c_reg = QuantumRegister(n, "c")
    d_reg = QuantumRegister(n, "d")
    temp = QuantumRegister(n, "temp")
    carry = QuantumRegister(1, "carry")
    iv_a_reg = QuantumRegister(n, "iv_a")
    iv_b_reg = QuantumRegister(n, "iv_b")
    iv_c_reg = QuantumRegister(n, "iv_c")
    iv_d_reg = QuantumRegister(n, "iv_d")
    anc = QuantumRegister(1, "anc")
    c_w = ClassicalRegister(w_size, "c_w")

    qc = QuantumCircuit(w_reg, a_reg, b_reg, c_reg, d_reg, temp, carry, iv_a_reg, iv_b_reg, iv_c_reg, iv_d_reg, anc, c_w)

    init_register(qc, n, a, a_reg)
    init_register(qc, n, b, b_reg)
    init_register(qc, n, c, c_reg)
    init_register(qc, n, d, d_reg)
    init_register(qc, n, a, iv_a_reg)
    init_register(qc, n, b, iv_b_reg)
    init_register(qc, n, c, iv_c_reg)
    init_register(qc, n, d, iv_d_reg)

    qc.x(anc)
    qc.h(anc)
    init_w(qc, w_reg)

    for _ in range(num_grover_iterations):
        oracle(qc, target_hash, w_size, n, a_reg, b_reg, c_reg, d_reg, w_reg, temp, carry, anc, iv_a_reg, iv_b_reg, iv_c_reg, iv_d_reg, iterations, k_list, s_list)
        diffusion(qc, w_reg)

    qc.measure(w_reg, c_w)
    return qc


def get_backend(backend_mode):
    if backend_mode == "simulator":
        return AerSimulator()
    service = QiskitRuntimeService()
    return service.least_busy(operational=True, simulator=False)


def run_trial(transpiled_qc, backend, backend_mode, shots, w_size, params, n, target):
    sampler = AerSampler() if backend_mode == "simulator" else Sampler(mode=backend)

    t0 = time.time()
    job = sampler.run([transpiled_qc], shots=shots)
    result = job.result()
    elapsed_ms = (time.time() - t0) * 1000

    counts = result[0].data.c_w.get_counts()

    found = False
    for bitstring, _ in counts.items():
        measured_val = int(bitstring.replace(" ", ""), 2)
        if tiny_md5_bits(n, w_size, w=measured_val, **params) == target:
            found = True
            break

    return elapsed_ms, found, counts


def run_benchmark(n, w_size, shots=3, num_trials=10, output="results_qiskit_real.json", backend_mode="real"):
    params = get_md5_params(n)
    max_val = (1 << w_size) - 1

    print(f"n={n}, w={w_size}, shots={shots}, trials={num_trials}, backend={backend_mode}")

    msg = random.randint(0, max_val)
    target = tiny_md5_bits(n, w_size, w=msg, **params)

    t0 = time.time()
    qc = grover_search(n, w_size, params["a"], params["b"], params["c"], params["d"], target, params["k_list"], params["s_list"], params["iterations"])
    build_time = time.time() - t0

    backend = get_backend(backend_mode)

    t0 = time.time()
    transpiled_qc = transpile(qc, backend, optimization_level=3)
    transpile_time = time.time() - t0
    print(f"Backend: {backend.name}, build={build_time:.1f}s, transpile={transpile_time:.1f}s, depth={transpiled_qc.depth()}")

    trial_times = []
    trial_found = []
    for trial_idx in range(num_trials):
        elapsed_ms, found, _ = run_trial(transpiled_qc, backend, backend_mode, shots, w_size, params, n, target)
        trial_times.append(elapsed_ms)
        trial_found.append(found)

    avg_ms = float(np.mean(trial_times))
    std_ms = float(np.std(trial_times))
    success_rate = sum(trial_found) / num_trials

    print(f"Success rate: {sum(trial_found)}/{num_trials}")

    result = {
        "n": n,
        "w_size": w_size,
        "shots_per_trial": shots,
        "num_trials": num_trials,
        "search_space": max_val + 1,
        "message": msg,
        "target_hash": target,
        "build_time_sec": build_time,
        "transpile_time_sec": transpile_time,
        "circuit_depth": qc.depth(),
        "circuit_gates": qc.size(),
        "transpiled_depth": transpiled_qc.depth(),
        "transpiled_gates": transpiled_qc.size(),
        "backend": backend.name,
        "trial_times_ms": trial_times,
        "avg_trial_time_ms": avg_ms,
        "std_trial_time_ms": std_ms,
        "min_trial_time_ms": float(np.min(trial_times)),
        "max_trial_time_ms": float(np.max(trial_times)),
        "trials_found": trial_found,
        "success_rate": success_rate,
        "backend_mode": backend_mode,
    }

    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=str, default="results_qiskit_real.json")
    parser.add_argument("--backend", choices=["real", "simulator"], default="simulator")
    args = parser.parse_args()

    run_benchmark(args.n, args.w, args.shots, args.trials, args.output, args.backend)


if __name__ == "__main__":
    main()
