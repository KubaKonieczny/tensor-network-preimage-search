import argparse
import json
import os
import pickle
import random
import time
import psutil
import numpy as np
import quimb.tensor as qtn

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


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024**2


def MAJ(circ, a, b, cin):
    circ.apply_gate("CX", a, b)
    circ.apply_gate("CX", a, cin)
    circ.apply_gate("CCX", cin, b, a)


def UMA(circ, a, b, cin):
    circ.apply_gate("CCX", cin, b, a)
    circ.apply_gate("CX", a, cin)
    circ.apply_gate("CX", cin, b)


def add_quantum(circ, n, target, added, ancilla):
    MAJ(circ, added[0], target[0], ancilla)
    for i in range(n - 1):
        MAJ(circ, added[i + 1], target[i + 1], added[i])
    for i in reversed(range(n - 1)):
        UMA(circ, added[i + 1], target[i + 1], added[i])
    UMA(circ, added[0], target[0], ancilla)


def add_k(circ, n, k, a, temp, ancilla):
    for i in range(n):
        if (k >> i) & 1:
            circ.apply_gate("X", temp[i])
    add_quantum(circ, n, a, temp, ancilla)
    for i in range(n):
        if (k >> i) & 1:
            circ.apply_gate("X", temp[i])


def F(circ, b, c, d):
    for i in range(len(b)):
        circ.apply_gate("CX", d[i], c[i])
        circ.apply_gate("CCX", b[i], c[i], d[i])


def F_uncompute(circ, b, c, d):
    for i in range(len(b)):
        circ.apply_gate("CCX", b[i], c[i], d[i])
        circ.apply_gate("CX", d[i], c[i])


def G(circ, b, c, d):
    for i in range(len(b)):
        circ.apply_gate("CX", c[i], b[i])
        circ.apply_gate("CCX", b[i], d[i], c[i])


def G_uncompute(circ, b, c, d):
    for i in range(len(b)):
        circ.apply_gate("CCX", b[i], d[i], c[i])
        circ.apply_gate("CX", c[i], b[i])


def H(circ, b, c, d):
    for i in range(len(b)):
        circ.apply_gate("CX", b[i], d[i])
        circ.apply_gate("CX", c[i], d[i])


def I(circ, b, c, d):
    for i in range(len(b)):
        circ.apply_gate("X", c[i])
        circ.apply_gate("X", b[i])
        circ.apply_gate("CCX", b[i], d[i], c[i])


def I_uncompute(circ, b, c, d):
    for i in reversed(range(len(b))):
        circ.apply_gate("CCX", b[i], d[i], c[i])
        circ.apply_gate("X", b[i])
        circ.apply_gate("X", c[i])


def shift(circ, a, s, n):
    s = s % n
    for _ in range(s):
        for i in reversed(range(len(a) - 1)):
            circ.apply_gate("SWAP", a[i], a[i + 1])


def swap_regs(circ, a, b, c, d):
    for i in range(len(a)):
        circ.apply_gate("SWAP", a[i], b[i])
        circ.apply_gate("SWAP", a[i], c[i])
        circ.apply_gate("SWAP", a[i], d[i])


def init_reg(circ, n, val, reg):
    for i in range(n):
        if (val >> i) & 1:
            circ.apply_gate("X", reg[i])


def init_w_superposition(circ, w):
    for q in w:
        circ.apply_gate("H", q)


def add_iv(circ, n, a, b, c, d, iv_a, iv_b, iv_c, iv_d, ancilla):
    add_quantum(circ, n, a, iv_a, ancilla)
    add_quantum(circ, n, b, iv_b, ancilla)
    add_quantum(circ, n, c, iv_c, ancilla)
    add_quantum(circ, n, d, iv_d, ancilla)


def md5_iteration_quantum(circ, it, n, w_size, a, b, c, d, w, temp, ancilla, k_list, s_list):
    num_chunks = w_size // n
    chunk_index = get_chunk_index(it, num_chunks)
    wq = [w[chunk_index * n + j] for j in range(n)]

    add_quantum(circ, n, a, wq, ancilla)

    if it < 16:
        F(circ, b, c, d)
        add_quantum(circ, n, a, d, ancilla)
        F_uncompute(circ, b, c, d)
    elif it < 32:
        G(circ, b, c, d)
        add_quantum(circ, n, a, c, ancilla)
        G_uncompute(circ, b, c, d)
    elif it < 48:
        H(circ, b, c, d)
        add_quantum(circ, n, a, d, ancilla)
        H(circ, b, c, d)
    else:
        I(circ, b, c, d)
        add_quantum(circ, n, a, c, ancilla)
        I_uncompute(circ, b, c, d)

    add_k(circ, n, k_list[it], a, temp, ancilla)
    shift(circ, a, s_list[it], n)
    add_quantum(circ, n, a, b, ancilla)
    swap_regs(circ, a, b, c, d)


def build_circuit(n, w_size, params, max_bond_build=None):
    a = list(range(0, n))
    b = list(range(n, 2 * n))
    c = list(range(2 * n, 3 * n))
    d = list(range(3 * n, 4 * n))
    w = list(range(4 * n, 4 * n + w_size))
    temp = list(range(4 * n + w_size, 5 * n + w_size))
    ancilla = 5 * n + w_size
    iv_a = list(range(5 * n + w_size + 1, 6 * n + w_size + 1))
    iv_b = list(range(6 * n + w_size + 1, 7 * n + w_size + 1))
    iv_c = list(range(7 * n + w_size + 1, 8 * n + w_size + 1))
    iv_d = list(range(8 * n + w_size + 1, 9 * n + w_size + 1))
    total = 9 * n + w_size + 1

    print(f"Total qubits: {total}")

    start = time.time()
    circ = qtn.CircuitMPS(total, max_bond=max_bond_build)

    init_reg(circ, n, params["a"], a)
    init_reg(circ, n, params["b"], b)
    init_reg(circ, n, params["c"], c)
    init_reg(circ, n, params["d"], d)
    init_reg(circ, n, params["a"], iv_a)
    init_reg(circ, n, params["b"], iv_b)
    init_reg(circ, n, params["c"], iv_c)
    init_reg(circ, n, params["d"], iv_d)

    init_w_superposition(circ, w)

    for i in range(params["iterations"]):
        md5_iteration_quantum(circ, i, n, w_size, a, b, c, d, w, temp, ancilla, params["k_list"], params["s_list"])

    add_iv(circ, n, a, b, c, d, iv_a, iv_b, iv_c, iv_d, ancilla)

    build_time = time.time() - start
    max_bond_after_build = max(circ.psi.bond_sizes())

    return circ, build_time, max_bond_after_build, (a, b, c, d, w)


def check_hash_quantum(psi, n, target, max_bond=None):
    P0 = np.array([[1, 0], [0, 0]])
    P1 = np.array([[0, 0], [0, 1]])

    mpo_list = [P1 if (target >> i) & 1 else P0 for i in range(4 * n)]
    mpo = qtn.MPO_product_operator(mpo_list)

    result = mpo.apply(psi.copy())
    if max_bond is not None:
        result.compress_all(max_bond=max_bond, inplace=True)
    return result


def reverse_bits(num, n_bits):
    result = 0
    for i in range(n_bits):
        if (num >> i) & 1:
            result |= 1 << (n_bits - 1 - i)
    return result


def query_circuit_full_dist(psi, n, target, qubits, params, max_bond=None):
    a, b, c, d, w = qubits
    w_size = len(w)
    w_start = 4 * n

    target_reversed = reverse_bits(target, 4 * n)
    result_mps = check_hash_quantum(psi, n, target_reversed, max_bond=max_bond)

    fixed = {}
    for i in range(4 * n):
        fixed[i] = (target_reversed >> i) & 1
    for i in range(4 * n + w_size, 5 * n + w_size):
        fixed[i] = 0
    fixed[5 * n + w_size] = 0
    for j in range(n):
        fixed[5 * n + w_size + 1 + j] = (params["a"] >> j) & 1
        fixed[6 * n + w_size + 1 + j] = (params["b"] >> j) & 1
        fixed[7 * n + w_size + 1 + j] = (params["c"] >> j) & 1
        fixed[8 * n + w_size + 1 + j] = (params["d"] >> j) & 1

    tn = result_mps.copy()
    for site, val in fixed.items():
        phys_ind = result_mps.site_ind(site)
        bra = np.zeros(2, dtype=complex)
        bra[val] = 1.0
        tn.add_tensor(qtn.Tensor(bra, inds=(phys_ind,), tags=(f"FIX{site}",)))

    w_inds = [result_mps.site_ind(i) for i in range(w_start, w_start + w_size)]
    contracted = tn.contract(output_inds=w_inds)
    psi_w = np.asarray(contracted.to_dense(w_inds)).flatten()

    probs = np.abs(psi_w) ** 2
    total_prob = float(probs.sum())

    probs /= total_prob
    probs = probs.reshape([2] * w_size)

    flat_probs = probs.reshape(-1)
    nz = np.where(flat_probs > 0.0)[0]
    nz_p = flat_probs[nz]
    w_vals = np.zeros(len(nz), dtype=np.int64)
    for j in range(w_size):
        w_vals |= ((nz >> (w_size - 1 - j)).astype(np.int64) & 1) << j
    order = np.argsort(nz_p)[::-1]
    dist = list(zip(w_vals[order].tolist(), nz_p[order].tolist()))
    return dist, total_prob


def query_circuit_sample(psi, n, target, qubits, n_samples=5000, top_k=1000, max_bond=None):
    a, b, c, d, w = qubits
    target_reversed = reverse_bits(target, 4 * n)

    result_mps = check_hash_quantum(psi, n, target_reversed, max_bond=max_bond)
    samples = list(result_mps.sample(n_samples))

    w_start = 4 * n
    best = {}
    for val, prob in samples:
        w_bits = [val[i] for i in range(w_start, w_start + len(w))]
        w_value = int("".join(map(str, reversed(w_bits))), 2)
        if w_value not in best or prob > best[w_value]:
            best[w_value] = prob

    return sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]


def run_mps_bench(n, w_size, mps_bond_values, max_bond_build=None, n_hashes=100, save_mps_dir=None, output_core=None, query_mode="full_dist", n_samples=5000, top_k=1000):
    params = get_md5_params(n)
    max_val = (1 << w_size) - 1
    num_targets = min(n_hashes, max_val + 1)

    messages = random.sample(range(max_val + 1), num_targets)
    targets = [tiny_md5_bits(n, w_size, w=m, **params) for m in messages]

    print(f"n={n}, w={w_size}, n_hashes={n_hashes}")

    mem_start = get_memory_mb()
    circ, build_time, max_bond_after_build, qubits = build_circuit(n, w_size, params, max_bond_build)
    mem_after_build = get_memory_mb()

    print(f"Build: {build_time:.1f}s, bond={max_bond_after_build}")

    original_psi = circ.psi.copy()
    results = []

    for mps_bond in mps_bond_values:
        label = str(mps_bond) if mps_bond is not None else "uncompressed"
        print(f"bond={label}")

        compressed_psi = original_psi.copy()
        compress_start = time.time()
        if mps_bond is not None:
            compressed_psi = compressed_psi.compress_all(max_bond=mps_bond)
        compress_time = time.time() - compress_start
        actual_bond = max(compressed_psi.bond_sizes())

        mps_file = None
        if save_mps_dir is not None:
            os.makedirs(save_mps_dir, exist_ok=True)
            mps_file = os.path.join(save_mps_dir, f"mps_n{n}_w{w_size}_bond{label}.pkl")
            with open(mps_file, "wb") as f:
                pickle.dump(compressed_psi, f)

        total_query_time = 0.0
        found_count = 0
        total_valid = 0
        per_query_results = []

        for msg, target in zip(messages, targets):
            query_start = time.time()
            if query_mode == "sample":
                dist = query_circuit_sample(compressed_psi, n, target, qubits, n_samples=n_samples, top_k=top_k, max_bond=mps_bond)
                hash_prob = None
            else:
                dist, hash_prob = query_circuit_full_dist(compressed_psi, n, target, qubits, params, max_bond=mps_bond)
            query_time = time.time() - query_start
            total_query_time += query_time

            valid_bond = [(w_val, float(prob)) for w_val, prob in dist if tiny_md5_bits(n, w_size, w=w_val, **params) == target]
            total_valid += len(valid_bond)
            if valid_bond:
                found_count += 1

            per_query_results.append(
                {
                    "message": msg,
                    "target": target,
                    "query_time_sec": query_time,
                    "hash_probability": hash_prob,
                    "distribution_size": len(dist),
                    "found": bool(valid_bond),
                    "valid_preimages": valid_bond,
                    "full_distribution": [(w_val, float(prob)) for w_val, prob in dist],
                }
            )

        mem_after_query = get_memory_mb()

        print(f"Query: {total_query_time:.1f}s, found={found_count}/{num_targets}")

        bond_result = {
            "n": n,
            "w_size": w_size,
            "max_bond_build": max_bond_build,
            "max_bond_after_build": max_bond_after_build,
            "mps_bond_target": mps_bond,
            "mps_bond_actual": actual_bond,
            "mps_file": mps_file,
            "build_time_sec": build_time,
            "compress_time_sec": compress_time,
            "query_time_sec": total_query_time,
            "query_mode": query_mode,
            "n_hashes": n_hashes,
            "num_queries": num_targets,
            "found": found_count,
            "success_rate": found_count / num_targets,
            "total_valid_preimages": total_valid,
            "time_per_query": total_query_time / num_targets,
            "memory_start_mb": mem_start,
            "memory_after_build_mb": mem_after_build,
            "memory_after_query_mb": mem_after_query,
            "per_query_results": per_query_results,
        }
        results.append(bond_result)

        if output_core is not None:
            bond_file = f"{output_core}_bond{label}.json"
            with open(bond_file, "w") as f:
                json.dump(bond_result, f, indent=2)

    return build_time, max_bond_after_build, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--max-bond-build", type=int, default=None)
    parser.add_argument("--mps-bonds", type=int, nargs="+", default=None)
    parser.add_argument("--include-uncompressed", action="store_true")
    parser.add_argument("--n-hashes", type=int, default=100)
    parser.add_argument("--save-mps-dir", type=str, default=None)
    parser.add_argument("--query-mode", choices=["full_dist", "sample"], default="full_dist")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results_mps.json")
    args = parser.parse_args()

    mps_bond_values = args.mps_bonds or []
    if args.include_uncompressed or not mps_bond_values:
        mps_bond_values = [None] + mps_bond_values

    print(f"n={args.n}, w={args.w}, query_mode={args.query_mode}, bonds={mps_bond_values}")

    output_core = args.output[:-5] if args.output.endswith(".json") else args.output

    build_time, max_bond_after_build, results = run_mps_bench(
        args.n,
        args.w,
        mps_bond_values,
        args.max_bond_build,
        args.n_hashes,
        save_mps_dir=args.save_mps_dir,
        output_core=output_core,
        query_mode=args.query_mode,
        n_samples=args.n_samples,
        top_k=args.top_k,
    )

    combined = {"config": vars(args), "build_time_sec": build_time, "max_bond_after_build": max_bond_after_build, "results": results}

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
