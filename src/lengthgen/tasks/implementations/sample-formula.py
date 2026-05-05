import random

# You can customize these:
VARIABLES   = ["true", "false"]  # Base atoms (true/false values)
UNARY_OPS   = ["¬"]
BINARY_OPS  = ["∧", "∨"]  # Removed → (implies)


def compute_formula_counts(max_n):
    """
    Compute f[n] = number of formulas of size n,
    for n = 0..max_n, under the grammar:
      F ::= variable | u F | (F o F)
    with size = number of nodes.
    """
    if max_n < 1:
        raise ValueError("max_n must be at least 1")

    f = [0] * (max_n + 1)
    k = len(VARIABLES)
    u = len(UNARY_OPS)
    b = len(BINARY_OPS)

    # Size 1: variables only
    f[1] = k

    # Sizes >= 2
    for n in range(2, max_n + 1):
        unary_count = u * f[n - 1]
        binary_sum = 0
        for i in range(1, n - 1):
            binary_sum += f[i] * f[n - 1 - i]
        binary_count = b * binary_sum
        f[n] = unary_count + binary_count

    return f


def sample_formula(n, counts, rng=None, natural=False, _unary_ops=None, _binary_ops=None):
    # Resolve operators once at the top-level call, then pass down
    if _unary_ops is None:
        _unary_ops = ["NOT"] if natural else UNARY_OPS
    if _binary_ops is None:
        _binary_ops = ["AND", "OR"] if natural else BINARY_OPS

    if rng is None:
        rng = random

    f = counts
    if n <= 0 or n >= len(f):
        raise ValueError("n must be between 1 and len(counts)-1")

    if n == 1:
        return rng.choice(VARIABLES)

    u = len(_unary_ops)
    b = len(_binary_ops)

    unary_count = u * f[n - 1]
    binary_pairs = sum(f[i] * f[n - 1 - i] for i in range(1, n - 1))
    binary_count = b * binary_pairs

    r = rng.randrange(unary_count + binary_count)

    if r < unary_count:
        op = rng.choice(_unary_ops)
        sub = sample_formula(n - 1, counts, rng, _unary_ops=_unary_ops, _binary_ops=_binary_ops)
        return op + sub

    r -= unary_count
    op_index = r % b
    pair_index = r // b

    for i in range(1, n - 1):
        block = f[i] * f[n - 1 - i]
        if pair_index < block:
            left  = sample_formula(i,         counts, rng, _unary_ops=_unary_ops, _binary_ops=_binary_ops)
            right = sample_formula(n - 1 - i, counts, rng, _unary_ops=_unary_ops, _binary_ops=_binary_ops)
            return f"({left} {_binary_ops[op_index]} {right})"
        pair_index -= block

    raise RuntimeError("Sampling logic error: did not find a split")


# --- Example usage ---
if __name__ == "__main__":
    max_size = 12
    counts = compute_formula_counts(max_size)

    size = 7
    print(f"Number of formulas of size {size}: {counts[size]}")
    print("Random samples of that exact size:")
    for _ in range(5):
        print("  ", sample_formula(size, counts))