import torch
import triton
from torch.nn import functional as F

def sequential_scan(
    h_t: torch.FloatTensor,
    A_bar: torch.FloatTensor,
    B_bar_x: torch.FloatTensor,
    C: torch.FloatTensor,
) -> torch.FloatTensor:
    y_ts = []
    for t in range(C.shape[-1]):
        h_t = (
            A_bar[..., t] * h_t + B_bar_x[..., t]
        )  # h_t = ̄A_t h_t + ̄Bx_t (eq. 2a - Mamba)
        y_t = torch.einsum(
            "bn,bdn->bd", C[..., t], h_t
        )  # y_t = C_t h_t (eq. 2b - Mamba)
        y_ts.append(y_t)

    return torch.stack(y_ts, dim=-1)


def parallel_scan(
    h_t: torch.FloatTensor,
    A_bar: torch.FloatTensor,
    B_bar_x: torch.FloatTensor,
    C: torch.FloatTensor,
    eps: float = 1e-8,
) -> torch.FloatTensor:
    """
    Parallel computation of the selective scan based on the algorithm described in
    "Efficient Parallelization of a Ubiquitous Sequential Computation". This method
    transforms the sequential operation `x_t = a_t * x_{t−1} + b_t` into a form
    suitable for parallel processing. It computes `log(x_t) = a_star_t + log(x_0 + b_star_t)`
    using parallel cumulative sums, where `a_star_t = cumsum(log(a_t))` and
    `b_star_t = logcumsumexp(log(b_t - a_star_t))`. This approach significantly
    enhances computational efficiency for large-scale tensor operations. In this method, we use the
    above formulation to compute `h_t = a_t * h_{t−1} + b_t` and `y_t = C_t h_t` in parallel.

    Reference:
    [Efficient Parallelization of a Ubiquitous Sequential Computation](https://arxiv.org/pdf/2311.06281.pdf)
    """
    # Combine B_bar_x and h_t into a single tensor
    B_bar_x = torch.cat([h_t[..., None], B_bar_x], dim=-1)  # [b x d x n x t + 1]

    # Compute the log of the parameters in complex space as log of negative numbers is complex
    log_A_bar = (
        A_bar.masked_fill(A_bar == 0, eps).to(torch.complex64)
    ).log()  # log a_t (eq. 2 - Efficient Parallelization)
    log_B_bar_x = (
        B_bar_x.masked_fill(B_bar_x == 0, eps).to(torch.complex64)
    ).log()  # log b_t (eq. 2 - Efficient Parallelization)

    # Compute the cumulative sum of log_A_bar along the time dimension
    A_bar_star = F.pad(
        torch.cumsum(log_A_bar, dim=-1), (1, 0)
    )  # a_star_t (eq. 2 - Efficient Parallelization) [b x d x n x t + 1]

    # Compute the log of the cumulative sum of log_B_bar_x along the time dimension
    B_bar_x_star = torch.logcumsumexp(
        log_B_bar_x - A_bar_star, dim=-1
    )  # b_star_t (eq. 2 - Efficient Parallelization) [b x d x n x t + 1]

    log_h_ts = (A_bar_star + B_bar_x_star)[
        ..., 1:
    ]  # log x_t (eq. 1 - Efficient Parallelization)
    h_ts = torch.exp(
        log_h_ts
    ).real  # x_t (eq. 3 - Efficient Parallelization) h_t = ̄A_t h_t + ̄Bx_t (eq. 2a - Mamba)

    return torch.einsum("bnt,bdnt->bdt", C, h_ts)  # y_t = C_t h_t  (eq. 2b - Mamba)


sequential_scan_jit = torch.jit.script(sequential_scan)
sequential_scan_compile = torch.compile(sequential_scan)
parallel_scan_jit = torch.jit.script(parallel_scan)
parallel_scan_compile = torch.compile(parallel_scan)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["length"],
        x_vals=[2**i for i in range(8, 10)],
        x_log=True,
        line_arg="implementation",
        line_vals=[
            "sequential_scan_eager",
            "sequential_scan_jit",
            "sequential_scan_compile",
            "parallel_scan_eager",
            "parallel_scan_jit",
            "parallel_scan_compile",
        ],
        line_names=[
            "Sequential Scan (Eager)",
            "Sequential Scan (JIT)",
            "Sequential Scan (Compile)",
            "Parallel Scan (Eager)",
            "Parallel Scan (JIT)",
            "Parallel Scan (Compile)",
        ],
        styles=[
            ("red", "-"),
            ("green", "-."),
            ("blue", "--"),
            ("black", "-"),
            ("orange", "-."),
            ("purple", "--"),
        ],
        ylabel="ms",
        plot_name="Scan",
        args={},
    )
)
def benchmark(length, implementation):
    h_t = torch.randn((2, 512, 64), device="cuda", dtype=torch.float32)
    A_bar = torch.randn((2, 512, 64, length), device="cuda", dtype=torch.float32)
    B_bar_x = torch.randn_like(A_bar)
    C = torch.randn((2, 64, length), device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if implementation == "sequential_scan_eager":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sequential_scan(h_t, A_bar, B_bar_x, C), quantiles=quantiles
        )
    if implementation == "sequential_scan_jit":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sequential_scan_jit(h_t, A_bar, B_bar_x, C), quantiles=quantiles
        )
    if implementation == "sequential_scan_compile":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sequential_scan_compile(h_t, A_bar, B_bar_x, C),
            quantiles=quantiles,
        )
    if implementation == "parallel_scan_eager":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: parallel_scan(h_t, A_bar, B_bar_x, C),
            quantiles=quantiles,
        )
    if implementation == "parallel_scan_jit":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: parallel_scan_jit(h_t, A_bar, B_bar_x, C), quantiles=quantiles
        )
    if implementation == "parallel_scan_compile":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: parallel_scan_compile(h_t, A_bar, B_bar_x, C),
            quantiles=quantiles,
        )

    return ms, min_ms, max_ms


benchmark.run(print_data=True, show_plots=True, save_path="runs")