"""MACE's `skip_tp` at deploy: a cuEq FullyConnectedTensorProduct of x with the one-hot element encoding
node_attrs (E x 0e) is a per-node, per-irrep linear map, out_l[n, m] = a[n] * x_l[n, m] @ W_l[z_n], with
z_n = argmax(node_attrs[n]) and a[n] = node_attrs[n, z_n] (0 for an all-zero padding row). A per-species GEMM
does the 1/E of the work the full 0e tensor product spends mostly multiplying zeros.

Weights are probed out of the module itself (no cuEq internal layout): one row per input channel u, x[row, off_in +
u] = 1 at m = 0, node_attrs = one_hot(z) gives W_l[z][u, v] = out[row, off_out + v], normalisation included.
Output irreps with no matching input irrep (MACE-MP-0 layer 0: 0e -> 0e+1o) stay zero, as upstream.

The Triton kernel takes BLOCK_M (node, m) rows in species-sorted order, loads them through the node permutation,
loops over the species present in the tile (sorted: usually one) and writes back to the original rows; the same
kernel with W^T gives the backward. Grid and permutation are device-side (argsort + gather: CUDA-graph capturable,
no host sync); autograd is first-order in x only (weights frozen, eval-only deploy).
"""
import os

import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32   # fixed tiles (autotune would break the CUDA-graph capture); 4070S sweep: BK 32 < 64, BN 128 ~ 64
INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'   # CPU kernel runs through the interpreter, else torch fallback


@triton.jit
def _species_mm(x_ptr, a_ptr, z_ptr, perm_ptr, w_ptr, out_ptr, n, dm, E, sx, so, K: tl.constexpr, N: tl.constexpr,
                BK: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr):
    """out[node, m, :] = a[node] * x[node, m, :] @ W[z[node], :, :] over sorted (node, m) rows of one irrep path;
    grid (row tiles, BN-column tiles), K in BK chunks (a whole 256 x 256 W tile overflows shared memory)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    ok = rows < n * dm
    nr = rows // dm
    m = rows - nr * dm
    node = tl.load(perm_ptr + nr, mask=ok, other=0)
    z = tl.load(z_ptr + node, mask=ok, other=0)
    a = tl.load(a_ptr + node, mask=ok, other=0.0)
    ar_k = tl.arange(0, BK)
    ar_v = tl.program_id(1) * BN + tl.arange(0, BN)
    src = node * sx + m * K
    acc = tl.zeros((BM, BN), tl.float32)
    todo = ok
    s = tl.min(tl.where(todo, z, E))
    while s < E:
        sel = todo & (z == s)
        for k0 in range(0, K, BK):
            kk = k0 + ar_k
            x = tl.load(x_ptr + src[:, None] + kk[None, :], mask=sel[:, None] & (kk[None, :] < K), other=0.0)
            w = tl.load(w_ptr + s * (K * N) + kk[:, None] * N + ar_v[None, :],
                        mask=(kk[:, None] < K) & (ar_v[None, :] < N), other=0.0)
            acc += tl.dot(x, w, input_precision='ieee')
        todo = todo & (z != s)
        s = tl.min(tl.where(todo, z, E))
    dst = node * so + m * N
    tl.store(out_ptr + dst[:, None] + ar_v[None, :], acc * a[:, None], mask=ok[:, None] & (ar_v[None, :] < N))


class _MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, z, perm, mod):
        ctx.save_for_backward(a, z, perm)
        ctx.mod = mod
        return mod._mm(x, a, z, perm, False)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        a, z, perm = ctx.saved_tensors
        return ctx.mod._mm(dy.contiguous(), a, z, perm, True), None, None, None, None


def _blocks(irreps):
    """[(mul, l, p)] of a cue Irreps (ir_mul block: off + m*mul + u)."""
    return [(mi.mul, mi.ir.l, mi.ir.p) for mi in irreps._mulirreps]


class SpeciesLinear(torch.nn.Module):
    """Drop-in for a cuEq skip_tp (x (N, irreps_in1) x one-hot node_attrs (N, E) -> irreps_out), eval-only.
    Per irrep path: W and W^T as (E, mul_in, mul_out) buffers; a path whose output irrep has no matching input
    irrep is absent, leaving its output columns zero."""

    @classmethod
    @torch.no_grad()   # leaf buffers: no probe graph kept alive (deepcopy / pickling)
    def from_skip_tp(cls, skip_tp):
        """Probe the module's weights: one row per input channel (m = 0), one block of rows per element class."""
        ins, outs = _blocks(skip_tp.irreps_in1), _blocks(skip_tp.irreps_out)
        dtype, device = skip_tp.weight.dtype, skip_tp.weight.device
        E = skip_tp.irreps_in2.num_irreps
        off_in = [sum(m * (2 * l + 1) for m, l, _ in ins[:i]) for i in range(len(ins))]
        off_out = [sum(m * (2 * l + 1) for m, l, _ in outs[:i]) for i in range(len(outs))]
        mp = max(m for m, _, _ in ins)
        u = torch.arange(mp, device=device)
        rows = torch.arange(E, device=device)[:, None] * mp + u[None, :]
        x = torch.zeros(E * mp, skip_tp.irreps_in1.dim, dtype=dtype, device=device)
        for off, (mul, _, _) in zip(off_in, ins):
            x[rows[:, :mul], off + u[:mul]] = 1.0
        na = torch.nn.functional.one_hot(torch.arange(E, device=device).repeat_interleave(mp), E).to(dtype)
        out = skip_tp(x, na).reshape(E, mp, skip_tp.irreps_out.dim)
        meta, ws, wts = [], [], []
        for oo, (mo, l, p) in zip(off_out, outs):
            hit = [(io, mi) for io, (mi, li, pi) in zip(off_in, ins) if (li, pi) == (l, p)]
            if not hit:
                continue   # unmatchable output irrep: zero, as in the tensor product with a 0e operand
            (io, mi), = hit
            w = out[:, :mi, oo:oo + mo].float().contiguous()
            meta.append((io, oo, 2 * l + 1, mi, mo))
            ws.append(w)
            wts.append(w.transpose(1, 2).contiguous())
        mod = cls(meta, E)
        mod.in_dim, mod.out_dim = skip_tp.irreps_in1.dim, skip_tp.irreps_out.dim
        for i, (w, wt) in enumerate(zip(ws, wts)):
            mod.register_buffer(f'w{i}', w, persistent=False)
            mod.register_buffer(f'wt{i}', wt, persistent=False)
        return mod

    def __init__(self, meta, n_elements):
        super().__init__()
        self.meta = meta   # (in offset, out offset, 2l+1, mul_in, mul_out) per path
        self.n_elements = n_elements

    def forward(self, x, node_attrs):
        z = node_attrs.argmax(1).int()
        a = node_attrs.gather(1, z.long()[:, None])[:, 0]
        return _MM.apply(x.contiguous(), a, z, torch.argsort(z, stable=True).int(), self)   # kernel assumes unit column stride

    def _mm(self, x, a, z, perm, transposed):
        """out = a * x @ W[z] per path (W^T when transposed: the backward). x (N, irreps_in1), out (N, irreps_out)."""
        out = x.new_zeros(x.shape[0], self.in_dim if transposed else self.out_dim)
        # per path: weight, x offset and width, out offset and width (transposed: dy -> dx, so the two swap)
        paths = []
        for i, (io, oo, d, mi, mo) in enumerate(self.meta):
            w = getattr(self, ('wt' if transposed else 'w') + str(i))
            xi, xw, oi, ow = (oo, mo, io, mi) if transposed else (io, mi, oo, mo)
            paths.append((w, xi, xw, oi, ow, d))
        if x.is_cuda != INTERPRET and x.dtype == torch.float32:   # fp32 CUDA kernel (CPU: interpreter); else eager fallback
            for w, xi, xw, oi, ow, d in paths:
                _species_mm[(triton.cdiv(x.shape[0] * d, BM), triton.cdiv(ow, BN))](
                    x[:, xi:], a, z, perm, w, out[:, oi:], x.shape[0], d, self.n_elements, x.stride(0),
                    out.stride(0), xw, ow, BK=BK, BN=BN, BM=BM, num_warps=4)
            return out
        # ponytail: CPU without the interpreter is the eager fallback (per-class matmul); deploy is CUDA only
        for w, xi, xw, oi, ow, d in paths:
            w = w.to(x.dtype)
            blk = out[:, oi:oi + d * ow]
            for s in torch.unique(z):
                m = z == s
                blk[m] = (x[m, xi:xi + d * xw].reshape(-1, d, xw) @ w[s]).reshape(-1, d * ow)
            out[:, oi:oi + d * ow] = blk * a[:, None]
        return out
