import argparse
import json
import random
import time
import psutil


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


def get_memory_mb():
    return psutil.Process().memory_info().rss / 1024**2


def F_norm(x, y, z, n):
    mask = (1 << n) - 1
    return ((x & y) | (~x & z)) & mask


def G_norm(x, y, z, n):
    mask = (1 << n) - 1
    return ((x & z) | (y & ~z)) & mask


def H_norm(x, y, z, n):
    mask = (1 << n) - 1
    return (x ^ y ^ z) & mask


def I_norm(x, y, z, n):
    mask = (1 << n) - 1
    return (y ^ (x | ~z)) & mask


def left_rotate_norm(value, shift, n):
    mask = (1 << n) - 1
    shift = shift % n
    return ((value << shift) | (value >> (n - shift))) & mask


def get_chunk_index(i, num_chunks):
    if i < 16:
        return i % num_chunks
    elif i < 32:
        return (1 + 5 * i) % num_chunks
    elif i < 48:
        return (5 + 3 * i) % num_chunks
    else:
        return (7 * i) % num_chunks


def tiny_md5_bits(n, w_size, a, b, c, d, w, k_list, s_list, iterations=64):
    mask_n = (1 << n) - 1
    num_chunks = w_size // n
    a0, b0, c0, d0 = a, b, c, d

    for i in range(iterations):
        chunk_index = get_chunk_index(i, num_chunks)
        w_chunk = (w >> (n * chunk_index)) & mask_n

        if i < 16:
            f_result = F_norm(b, c, d, n)
        elif i < 32:
            f_result = G_norm(b, c, d, n)
        elif i < 48:
            f_result = H_norm(b, c, d, n)
        else:
            f_result = I_norm(b, c, d, n)

        temp = (a + f_result + w_chunk + k_list[i]) & mask_n
        temp = left_rotate_norm(temp, s_list[i], n)
        temp = (temp + b) & mask_n
        a, b, c, d = d, temp, b, c

    a = (a + a0) & mask_n
    b = (b + b0) & mask_n
    c = (c + c0) & mask_n
    d = (d + d0) & mask_n

    return (a << (3 * n)) | (b << (2 * n)) | (c << n) | d


def brute_force_single(target_hash, n, w_size, params):
    max_val = (1 << w_size) - 1
    start = time.time()

    for candidate in range(max_val + 1):
        h = tiny_md5_bits(n, w_size, **params, w=candidate)
        if h == target_hash:
            return candidate, candidate + 1, time.time() - start

    return -1, max_val + 1, time.time() - start


def brute_force_multiple(target_hashes, n, w_size, params):
    max_val = (1 << w_size) - 1
    target_set = set(target_hashes)
    found = {}

    start = time.time()

    for candidate in range(max_val + 1):
        h = tiny_md5_bits(n, w_size, **params, w=candidate)
        if h in target_set and h not in found:
            found[h] = candidate
            if len(found) == len(target_hashes):
                break

    return found, candidate + 1, time.time() - start


def get_md5_params(n):
    mask = (1 << n) - 1
    k_list = [(k >> (32 - n)) & mask for k in MD5_K]
    s_list = [max(1, min(n - 1, round(s * n / 32))) if n > 1 else 1 for s in MD5_S]
    return {
        "a": 0x67452301 & mask,
        "b": 0xEFCDAB89 & mask,
        "c": 0x98BADCFE & mask,
        "d": 0x10325476 & mask,
        "k_list": k_list,
        "s_list": s_list,
        "iterations": 64,
    }


def run_benchmark(n, w_size, mode, target_pct=0.05):
    params = get_md5_params(n)
    max_val = (1 << w_size) - 1
    mem_start = get_memory_mb()


    if mode == "single":
        msg = random.randint(0, max_val)
        target = tiny_md5_bits(n, w_size, **params, w=msg)

        found_msg, attempts, elapsed = brute_force_single(target, n, w_size, params)

        mem_end = get_memory_mb()
        result = {
            "n": n,
            "w_size": w_size,
            "mode": mode,
            "search_space": max_val + 1,
            "params": params,
            "time_sec": elapsed,
            "attempts": attempts,
            "speed_ops_sec": attempts / elapsed if elapsed > 0 else 0,
            "found": found_msg != -1,
            "memory_start_mb": mem_start,
            "memory_end_mb": mem_end,
            "memory_delta_mb": mem_end - mem_start,
        }

        print(f"time={elapsed:.2f}s, found={found_msg != -1}")

    else:
        num_targets = max(1, int((max_val + 1) * target_pct))
        messages = random.sample(range(max_val + 1), num_targets)
        targets = [tiny_md5_bits(n, w_size, **params, w=m) for m in messages]

        found, attempts, elapsed = brute_force_multiple(targets, n, w_size, params)

        mem_end = get_memory_mb()
        num_unique_targets = len(set(targets))
        result = {
            "n": n,
            "w_size": w_size,
            "mode": mode,
            "search_space": max_val + 1,
            "params": params,
            "num_targets": num_targets,
            "num_unique_targets": num_unique_targets,
            "target_percentage": target_pct,
            "time_sec": elapsed,
            "attempts": attempts,
            "speed_ops_sec": attempts / elapsed if elapsed > 0 else 0,
            "found": len(found),
            "success_rate": len(found) / num_unique_targets,
            "memory_start_mb": mem_start,
            "memory_end_mb": mem_end,
            "memory_delta_mb": mem_end - mem_start,
        }

        print(f"time={elapsed:.2f}s, found={len(found)}/{num_unique_targets}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", required=True)
    parser.add_argument("--w", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=["single", "multiple", "both"], default="both")
    parser.add_argument("--target-pct", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="results_bruteforce.json")
    args = parser.parse_args()


    modes = ["single", "multiple"] if args.mode == "both" else [args.mode]

    print(f"n={args.n}, w={args.w}, modes={modes}")

    results = []
    for i, n in enumerate(args.n):
        for mode in modes:
            result = run_benchmark(n, args.w[i], mode, args.target_pct)
            results.append(result)

    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
