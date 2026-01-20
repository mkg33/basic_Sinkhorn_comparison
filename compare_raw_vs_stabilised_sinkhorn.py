import os
import time
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "lines.antialiased": True,
        "patch.antialiased": True,
        "text.antialiased": True,
    }
)

graphs = [
    {"kind": "grid", "side": 8, "periodic": False},
    {"kind": "grid", "side": 10, "periodic": True},
    {"kind": "grid", "side": 14, "periodic": True},
    {"kind": "torus3d", "side": 4},
    {"kind": "hypercube", "dim": 6},
    {"kind": "ladder", "length": 30},
    {"kind": "path", "n": 60},
    {"kind": "cycle", "n": 30},
    {"kind": "star", "n": 40},
    {"kind": "complete", "n": 14},
    {"kind": "barbell", "clique_size": 10, "bridge_length": 3},
    {"kind": "grid_bridge", "side": 6, "bridge_length": 4},
    {"kind": "broom", "path_length": 30, "star_size": 10},
    {"kind": "binary_tree", "depth": 5},
    {"kind": "random_regular", "n": 120, "d": 3, "seed": 1},
    {"kind": "lollipop", "clique_size": 12, "path_length": 30},
]

side = 10
grid_periodic = False
ts = [0.5, 1.0, 2.0]
BETA = 2.0
ETA = 1e-2

etas = None

# the no. of iterations is arbitrary, of course
max_it = 4000000
# this precision is not too loose
eps = 1e-7


marg_mode = "random"
seed = 0
use_deg = False
out_root = "plots"
print_mode = "simple"

# full results
make_plots = True
save_csv = True
clean_outputs = True
max_tries = 200
small = float(np.finfo(float).tiny)

# extra precision
rtol = 1e-12
atol = 1e-15


def hd(x, y):
    x = np.maximum(x, small)
    y = np.maximum(y, small)
    lr = np.log(x) - np.log(y)
    return float(np.max(lr) - np.min(lr))


def proj(m):
    logm = np.log(np.maximum(m, small))
    n = m.shape[0]
    delta = 0.0
    for i in range(n):
        for k in range(n):
            v = logm[i] - logm[k]
            delta = max(delta, float(v.max() - v.min()))
    return delta


def contr(delta):
    d = np.longdouble(delta)
    t = np.exp(-d / 2)
    q = (1 - t) / (1 + t)
    rho = q * q
    one_m = (4 * t) / ((1 + t) * (1 + t))
    return q, rho, one_m


def osc(m, k):
    m = np.maximum(m, small)
    k = np.maximum(k, small)
    lr = np.log(m) - np.log(k)
    return float(np.max(lr) - np.min(lr))

# expm works OK
def heat(t, q):
    a = np.real(expm(t * q))
    a = np.maximum(a, small)
    rs = a.sum(axis=1, keepdims=True)
    if not np.allclose(rs, 1.0, rtol=rtol, atol=atol):
        a = a / rs
    return a

# naive Sinkhorn implementation for the basic comparison
def sinkhorn(m, r, c, b0):
    delta = proj(m)
    q, rho, one_m = contr(delta)
    tau = float(one_m * np.longdouble(eps))

    b = b0 / b0.sum()
    conv = False
    last_h = float("inf")

    for it in range(int(max_it)):
        mb = m @ b
        if not np.all(np.isfinite(mb)) or np.any(mb <= 0.0):
            break
        a = r / mb
        mt_a = m.transpose() @ a
        if not np.all(np.isfinite(mt_a)) or np.any(mt_a <= 0.0):
            break
        b1 = c / mt_a
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b1)):
            break

        b1 = b1 / b1.sum()
        last_h = hd(b, b1)
        if last_h <= tau:
            conv = True
            b = b1
            break
        b = b1

    mb = m @ b
    if not np.all(np.isfinite(mb)) or np.any(mb <= 0.0) or not np.all(np.isfinite(b)):
        a = np.full_like(b, np.nan)
        resid = float("inf")
    else:
        a = r / mb
        row_sums = a * (m @ b)
        col_sums = b * (m.transpose() @ a)
        resid = float(max(np.max(np.abs(row_sums - r)), np.max(np.abs(col_sums - c))))

    return {
        "a": a,
        "b": b,
        "iters": it + 1,
        "delta": float(delta),
        "q": float(q),
        "q_ld": q,
        "rho": float(rho),
        "one_minus_rho": float(one_m),
        "tau": float(tau),
        "resid": float(resid),
        "hilbert_resid": float(last_h),
        "converged": bool(conv),
    }


def clean(path):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if name.endswith((".png", ".pdf", ".svg")) or name in ("results.csv", "results_all.csv"):
            os.remove(os.path.join(path, name))


def save(fig, base):
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.03)


def marg(n, mode="uniform", seed=0):
    if mode == "uniform":
        r = np.full(n, 1.0 / n, dtype=float)
        c = np.full(n, 1.0 / n, dtype=float)
        return r, c
    rng = np.random.default_rng(seed)
    r = rng.random(n)
    c = rng.random(n)
    r /= r.sum()
    c /= c.sum()
    return r, c


def dist_all(w):
    n = w.shape[0]
    neighbours = [np.flatnonzero(w[i]).tolist() for i in range(n)]
    dist = np.full((n, n), np.inf, dtype=float)
    for s in range(n):
        dist[s, s] = 0.0
        q = deque([s])
        while q:
            v = q.popleft()
            dv = dist[s, v]
            for u in neighbours[v]:
                if dist[s, u] == np.inf:
                    dist[s, u] = dv + 1.0
                    q.append(u)
    return dist

# this probably won't work but the future plots should have better overlaps if we get more data
def overlap(df, key, rtol=1e-6, atol=1e-12):
    methods = list(df["method"].unique())
    series = {}
    for m in methods:
        sub = df[df["method"] == m].sort_values("t")
        series[m] = (sub["t"].to_numpy(), sub[key].to_numpy())
    out = {m: set() for m in methods}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1 = methods[i]
            m2 = methods[j]
            x1, y1 = series[m1]
            x2, y2 = series[m2]
            if len(x1) != len(x2) or not np.allclose(x1, x2, rtol=0.0, atol=0.0):
                continue
            mask = np.isfinite(y1) & np.isfinite(y2)
            if mask.any() and np.allclose(y1[mask], y2[mask], rtol=rtol, atol=atol):
                out[m1].add(m2)
                out[m2].add(m1)
    return out


def grid(side, periodic=False):
    n = side * side
    w = np.zeros((n, n), dtype=float)

    def idx(i, j):
        return i * side + j

    for i in range(side):
        for j in range(side):
            u = idx(i, j)

            ni = i + 1
            if periodic:
                ni %= side
            if ni < side:
                v = idx(ni, j)
                if v != u:
                    w[u, v] = 1.0
                    w[v, u] = 1.0

            nj = j + 1
            if periodic:
                nj %= side
            if nj < side:
                v = idx(i, nj)
                if v != u:
                    w[u, v] = 1.0
                    w[v, u] = 1.0

    label = f"{'torus' if periodic else 'grid'}{side}"
    return w, label


def lolli(m, l):
    m = int(m)
    l = int(l)
    n = m + l
    w = np.zeros((n, n), dtype=float)

    for i in range(m):
        for j in range(i + 1, m):
            w[i, j] = 1.0
            w[j, i] = 1.0

    for t in range(m, m + l - 1):
        w[t, t + 1] = 1.0
        w[t + 1, t] = 1.0

    w[0, m] = 1.0
    w[m, 0] = 1.0

    label = f"lollipop_m{m}_l{l}"
    return w, label


def pathg(n):
    n = int(n)
    w = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        w[i, i + 1] = 1.0
        w[i + 1, i] = 1.0
    label = f"path{n}"
    return w, label


def cyc(n):
    n = int(n)
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        w[i, j] = 1.0
        w[j, i] = 1.0
    label = f"cycle{n}"
    return w, label


def star(n):
    n = int(n)
    w = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        w[0, i] = 1.0
        w[i, 0] = 1.0
    label = f"star{n}"
    return w, label


def comp(n):
    n = int(n)
    w = np.ones((n, n), dtype=float)
    np.fill_diagonal(w, 0.0)
    label = f"complete{n}"
    return w, label


def conn(w):
    n = w.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        v = stack.pop()
        for u in np.flatnonzero(w[v]):
            if not seen[u]:
                seen[u] = True
                stack.append(u)
    return bool(np.all(seen))


def lad(length):
    length = int(length)
    n = 2 * length
    w = np.zeros((n, n), dtype=float)
    for i in range(length):
        a = i
        b = i + length
        w[a, b] = 1.0
        w[b, a] = 1.0
        if i + 1 < length:
            w[a, a + 1] = 1.0
            w[a + 1, a] = 1.0
            w[b, b + 1] = 1.0
            w[b + 1, b] = 1.0
    label = f"ladder{length}"
    return w, label

# TODO: fix, done
def torus3(side):
    side = int(side)
    n = side * side * side
    w = np.zeros((n, n), dtype=float)

    def idx(i, j, k):
        return (i * side + j) * side + k

    for i in range(side):
        for j in range(side):
            for k in range(side):
                u = idx(i, j, k)
                ni = (i + 1) % side
                nj = (j + 1) % side
                nk = (k + 1) % side
                v = idx(ni, j, k)
                w[u, v] = 1.0
                w[v, u] = 1.0
                v = idx(i, nj, k)
                w[u, v] = 1.0
                w[v, u] = 1.0
                v = idx(i, j, nk)
                w[u, v] = 1.0
                w[v, u] = 1.0
    label = f"torus3d{side}"
    return w, label


def cube(dim):
    dim = int(dim)
    n = 1 << dim
    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for bit in range(dim):
            j = i ^ (1 << bit)
            w[i, j] = 1.0
            w[j, i] = 1.0
    label = f"hypercube{dim}"
    return w, label


def barbell(m, bridge_len):
    m = int(m)
    bridge_len = int(bridge_len)
    internal = bridge_len - 1
    n = 2 * m + internal
    w = np.zeros((n, n), dtype=float)

    for i in range(m):
        for j in range(i + 1, m):
            w[i, j] = 1.0
            w[j, i] = 1.0

    right_offset = m + internal
    for i in range(right_offset, right_offset + m):
        for j in range(i + 1, right_offset + m):
            w[i, j] = 1.0
            w[j, i] = 1.0

    left_anchor = 0
    right_anchor = right_offset
    path_nodes = [left_anchor] + list(range(m, m + internal)) + [right_anchor]
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        w[u, v] = 1.0
        w[v, u] = 1.0

    label = f"barbell_m{m}_b{bridge_len}"
    return w, label


def gridbridge(side, bridge_length):
    side = int(side)
    bridge_length = int(bridge_length)
    w1, _ = grid(side, periodic=False)
    w2, _ = grid(side, periodic=False)
    n1 = w1.shape[0]
    n2 = w2.shape[0]
    internal = bridge_length - 1
    n = n1 + internal + n2
    w = np.zeros((n, n), dtype=float)
    w[:n1, :n1] = w1
    w[n1 + internal : n1 + internal + n2, n1 + internal : n1 + internal + n2] = w2

    left_anchor = n1 - 1
    right_anchor = n1 + internal
    path_nodes = [left_anchor] + list(range(n1, n1 + internal)) + [right_anchor]
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        w[u, v] = 1.0
        w[v, u] = 1.0

    label = f"gridbridge{side}_b{bridge_length}"
    return w, label


def broom(path_len, star_size):
    path_len = int(path_len)
    star_size = int(star_size)
    n = path_len + star_size
    w = np.zeros((n, n), dtype=float)
    for i in range(path_len - 1):
        w[i, i + 1] = 1.0
        w[i + 1, i] = 1.0
    hub = 0
    for i in range(path_len, n):
        w[hub, i] = 1.0
        w[i, hub] = 1.0
    label = f"broom_p{path_len}_s{star_size}"
    return w, label


def tree(depth):
    depth = int(depth)
    n = (1 << (depth + 1)) - 1
    w = np.zeros((n, n), dtype=float)
    for i in range((1 << depth) - 1):
        left = 2 * i + 1
        right = 2 * i + 2
        w[i, left] = 1.0
        w[left, i] = 1.0
        w[i, right] = 1.0
        w[right, i] = 1.0
    label = f"binarytree{depth}"
    return w, label


def randreg(n, d, seed=None, max_tries=200):
    n = int(n)
    d = int(d)
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n), dtype=float)
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for i in range(0, len(stubs), 2):
            u = int(stubs[i])
            v = int(stubs[i + 1])
            if u == v:
                ok = False
                break
            a = u if u < v else v
            b = v if u < v else u
            if (a, b) in edges:
                ok = False
                break
            edges.add((a, b))
        if not ok:
            continue
        w = np.zeros((n, n), dtype=float)
        for a, b in edges:
            w[a, b] = 1.0
            w[b, a] = 1.0
        if conn(w):
            label = f"random_regular_n{n}_d{d}"
            return w, label
    label = f"random_regular_n{n}_d{d}"
    return w, label

# general runner
def run(q, dist, mu, r, c, label):
    n = mu.shape[0]
    methods = ["raw", "constant", "stabilised"]
    b0 = {name: np.full(n, 1.0 / n, dtype=float) for name in methods}
    rows = []

    if print_mode == "csv":
        print(
            "graph,n,stage,t,eta,method,iterations,runtime_s,"
            "final_residual,final_residual_hilbert,one_minus_rho,"
            "bias_emp_to_raw,e_to_raw"
        )

    for stage, t in enumerate(ts, start=1):
        t = float(t)
        eta = float(etas[stage - 1]) if etas is not None else float(ETA)

        a = heat(t, q)
        rcut = float(BETA) * float(np.sqrt(t))
        atr = a * (dist <= rcut)

        radius = float(np.sqrt(t))
        vols = (dist <= radius) @ mu
        u = 1.0 / vols

        ones = np.ones(n, dtype=float)
        mc = atr + eta * (ones[:, None] * ones[None, :])
        ms = atr + eta * (u[:, None] * mu[None, :])
        kernels = [("raw", a), ("constant", mc), ("stabilised", ms)]

        if print_mode == "simple":
            print(f"\nt = {t:g}   (eta = {eta:g})")

        raw_b = None
        raw_m = a

        for name, m in kernels:
            start = time.perf_counter()
            out = sinkhorn(m, r, c, b0[name])
            runtime = time.perf_counter() - start
            b0[name] = out["b"]

            if name == "raw":
                raw_b = out["b"]
                e_to_raw = 0.0
                bias_emp = 0.0
            else:
                e_to_raw = osc(m, raw_m)
                bias_emp = hd(out["b"], raw_b) if raw_b is not None else float("nan")

            if print_mode == "simple":
                print(
                    name,
                    "iters =",
                    out["iters"],
                    "q =",
                    round(out["q"], 6),
                    "rho =",
                    round(out["rho"], 6),
                    "Delta =",
                    round(out["delta"], 6),
                    "hilb_resid =",
                    f"{out['hilbert_resid']:.3e}",
                    "marg_resid =",
                    f"{out['resid']:.3e}",
                    "bias_dH_to_raw =",
                    (f"{bias_emp:.3e}" if np.isfinite(bias_emp) else "nan"),
                )

            if print_mode == "csv":
                print(
                    f"{label},{n},{stage},{t:g},{eta:g},{name},"
                    f"{out['iters']},"
                    f"{runtime:.6g},"
                    f"{out['resid']:.6g},{out['hilbert_resid']:.6g},"
                    f"{out['one_minus_rho']:.6g},"
                    f"{bias_emp:.6g},{e_to_raw:.6g}"
                )

            rows.append(
                {
                    "graph": label,
                    "n": n,
                    "stage": stage,
                    "t": t,
                    "eta": eta,
                    "method": name,
                    "iters": out["iters"],
                    "one_minus_rho": out["one_minus_rho"],
                    "resid": out["resid"],
                    "hilbert_resid": out["hilbert_resid"],
                    "runtime_s": runtime,
                    "e_to_raw": e_to_raw,
                    "bias_emp_to_raw": bias_emp,
                }
            )

    return pd.DataFrame(rows)

# TODO: change to colourblind, OK
# TODO replace lines by scatter, OK
def plot(df, y, ylabel, logy=False):
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    over = overlap(df, y)
    order = ["raw", "constant", "stabilised"]
    methods = [m for m in order if m in set(df["method"])]
    marks = {"raw": "o", "constant": "s", "stabilised": "^"}
    cols = {"raw": "#000000", "constant": "#E69F00", "stabilised": "#0072B2"}
    cap_met = []
    if y == "iters":
        for method in methods:
            sub = df[df["method"] == method]
            if (sub["iters"] >= max_it).any():
                cap_met.append(method)
    for method in methods:
        sub = df[df["method"] == method].sort_values("t")
        label = method if not over[method] else f"{method} (overlap)"
        if y == "iters" and method in cap_met:
            label = f"{label} (cap)"
        ax.scatter(
            sub["t"],
            sub[y],
            marker=marks.get(method, "o"),
            label=label,
            color=cols.get(method, "#000000"),
            s=60,
            linewidths=0.8,
            edgecolors="#000000",
            alpha=0.9,
        )
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    if not df.empty:
        gname = str(df["graph"].iloc[0])
        ax.set_title(f"{gname}: raw vs constant floor vs stabilised")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    if y == "iters" and cap_met:
        cap_text = f"cap hit: {', '.join(cap_met)} (cap={max_it})"
        ax.text(0.02, 0.98, cap_text, transform=ax.transAxes, va="top", ha="left")
    return fig


def one(w, label):
    n = w.shape[0]
    if use_deg:
        mu = w.sum(axis=1)
    else:
        mu = np.ones(n, dtype=float)

    q = w / mu[:, None]
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))

    dist = dist_all(w)
    r, c = marg(n, mode=marg_mode, seed=seed)

    df = run(q, dist, mu, r, c, label)

    out_dir = os.path.join(out_root, label)
    if save_csv or make_plots:
        os.makedirs(out_dir, exist_ok=True)
        if clean_outputs:
            clean(out_dir)

    if save_csv:
        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    if make_plots and not df.empty:
        figs = [
            ("iters", "iterations", False),
            ("one_minus_rho", "1 - rho  (rho = q^2)", True),
            ("bias_emp_to_raw", "dH(b_method, b_raw)  (same t)", False),
        ]
        for key, ylabel, logy in figs:
            fig = plot(df, key, ylabel, logy=logy)
            save(fig, os.path.join(out_dir, key))
            plt.close(fig)

    return df


def main():
    os.makedirs(out_root, exist_ok=True)
    if clean_outputs:
        clean(out_root)
        for name in os.listdir(out_root):
            path = os.path.join(out_root, name)
            if os.path.isdir(path):
                clean(path)

    if graphs is None:
        w, label = grid(side, periodic=grid_periodic)
        df_all = one(w, label)
        if save_csv:
            df_all.to_csv(os.path.join(out_root, "results_all.csv"), index=False)
        return

    dfs = []
    for spec in graphs:
        kind = spec.get("kind", None)
        if kind == "grid":
            w, label = grid(int(spec["side"]), periodic=bool(spec.get("periodic", False)))
        elif kind == "torus3d":
            w, label = torus3(int(spec["side"]))
        elif kind == "hypercube":
            w, label = cube(int(spec["dim"]))
        elif kind == "ladder":
            w, label = lad(int(spec["length"]))
        elif kind == "path":
            w, label = pathg(int(spec["n"]))
        elif kind == "cycle":
            w, label = cyc(int(spec["n"]))
        elif kind == "star":
            w, label = star(int(spec["n"]))
        elif kind == "complete":
            w, label = comp(int(spec["n"]))
        elif kind == "barbell":
            w, label = barbell(int(spec["clique_size"]), int(spec["bridge_length"]))
        elif kind == "grid_bridge":
            w, label = gridbridge(int(spec["side"]), int(spec["bridge_length"]))
        elif kind == "broom":
            w, label = broom(int(spec["path_length"]), int(spec["star_size"]))
        elif kind == "binary_tree":
            w, label = tree(int(spec["depth"]))
        elif kind == "random_regular":
            w, label = randreg(
                int(spec["n"]),
                int(spec["d"]),
                seed=spec.get("seed", seed),
                max_tries=max_tries,
            )
        elif kind == "lollipop":
            w, label = lolli(int(spec["clique_size"]), int(spec["path_length"]))
        else:
            continue

        df = one(w, label)
        dfs.append(df)

    if dfs and save_csv:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(os.path.join(out_root, "results_all.csv"), index=False)

# magic
if __name__ == "__main__":
    main()
