#!/usr/bin/python3
#!/usr/bin/env python3
"""compute_optimal_p_send.py

Find the value of *p_send* that maximises the expected number of
messages delivered per simulation step and estimate throughput per
second when each robot broadcasts with a given frequency (Hz).

Definitions
===========
    p_received   = 1 / (1 + α · msg_size**β · p_send**γ · cluster_size**δ)
    msg_per_step = cluster_size · p_send · p_received

Optimum
=======
    p_send_opt = [ 1 / (α · msg_size**β · cluster_size**δ · (γ − 1)) ] ** (1/γ)

Behaviour
=========
* By **default** the optimum is clipped to the probability range
  0 ≤ p_send ≤ 1.  Use ``--no-clip`` / ``-u`` to allow larger values.
* Throughput metrics printed:
    - Optimal ``p_send``
    - Probability a single message is received (``p_received``)
    - Expected correctly‐received messages **per step**
    - Messages **sent per second**
    - Messages **correctly received per second**
"""
from __future__ import annotations

import argparse
from typing import Tuple


def compute_optimal_p_send(
    cluster_size: float,
    msg_size: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    clip: bool = True,
) -> Tuple[float, float, float]:
    """Return ``(p_send_opt, msg_per_step_opt, p_received)``.

    ``p_send_opt``
        Optimal send probability per simulation step.
    ``msg_per_step_opt``
        Expected correctly‐received messages per step.
    ``p_received``
        Probability a single message is correctly received.
    """

    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 for a finite optimum")

    c_const = alpha * (msg_size ** beta) * (cluster_size ** delta)
    p_send_opt = (1.0 / (c_const * (gamma - 1.0))) ** (1.0 / gamma)

    if clip:
        p_send_opt = max(0.0, min(1.0, p_send_opt))

    p_received = 1.0 / (1.0 + c_const * (p_send_opt ** gamma))
    msg_per_step_opt = cluster_size * p_send_opt * p_received

    return p_send_opt, msg_per_step_opt, p_received


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the optimal p_send and throughput estimates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-c", "--cluster_size", type=float, required=True,
                        help="Cluster size (n)")
    parser.add_argument("-m", "--msg_size", type=float, required=True,
                        help="Message size in bytes (m)")
    parser.add_argument("-a", "--alpha", type=float, default=0.000004,
                        help="Coefficient α")
    parser.add_argument("-b", "--beta", type=float, default=2.8096,
                        help="Exponent β")
    parser.add_argument("-g", "--gamma", type=float, default=2.3807,
                        help="Exponent γ (must be > 1)")
    parser.add_argument("-d", "--delta", type=float, default=1.2457,
                        help="Exponent δ")

    parser.add_argument("-f", "--frequency", type=float, default=30,
                        help="Sending frequency per robot in Hz (steps per second)")

    parser.add_argument("--no-clip", "-u", dest="clip", action="store_false",
                        help="Allow p_send to exceed the probability range [0,1]")
    parser.set_defaults(clip=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.frequency <= 0.0:
        raise ValueError("frequency must be positive")

    p_opt, mps_opt, p_received = compute_optimal_p_send(
        cluster_size=args.cluster_size,
        msg_size=args.msg_size,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        clip=args.clip,
    )

    msgs_sent_per_sec = args.cluster_size * p_opt * args.frequency
    msgs_recv_per_sec = mps_opt * args.frequency

    print(f"Optimal p_send: {p_opt:.6g}")
    print(f"Probability message received (p_received): {p_received:.6g}")
    print(f"Expected messages per step (received): {mps_opt:.6g}")
    print(f"Messages sent per second: {msgs_sent_per_sec:.6g}")
    print(f"Messages correctly received per second: {msgs_recv_per_sec:.6g}")

    if args.clip and p_opt in (0.0, 1.0):
        print("(Optimum lies at boundary due to clipping)")


if __name__ == "__main__":
    main()

