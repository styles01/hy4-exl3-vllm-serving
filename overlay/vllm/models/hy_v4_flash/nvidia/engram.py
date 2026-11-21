# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash frozen n-gram embeddings with TP head sharding and V2 lookback."""

import os

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import NgramHashState
from vllm.models.qwen4_exp.common.ngram_embedding import (
    _lookup_ple_embedding_from_pinned_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        self.lookup_table = None
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("engram_pad_id is outside the vocabulary")
        groups = config.engram_layer_head_vocab_sizes
        self.primes = tuple(
            groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids
        )
        flat = [[p for order in group for p in order] for group in self.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )
        if any(sum(sizes) > np.iinfo(np.int32).max for sizes in flat):
            raise ValueError("Engram flattened bucket offsets must fit int32")

    def load_lookup(self, value):
        if (
            value.dtype not in (torch.int32, torch.int64)
            or value.ndim != 1
            or not 0 < value.numel() <= self.vocab_size
        ):
            raise ValueError("Invalid frozen Engram token map")
        lookup = value.detach().cpu().numpy()
        if np.any(lookup < 0) or np.any(lookup > np.iinfo(np.int32).max):
            raise ValueError("Invalid frozen Engram token map")
        if self.lookup_table is not None:
            if not np.array_equal(self.lookup_table, lookup):
                raise ValueError("Engram layers must share the same token map")
            return False
        self.compressed_vocab_size = int(lookup.max()) + 1
        self.lookup_table = lookup.copy()
        self.token_map = lookup.tolist()
        self.token_map.extend(range(len(self.token_map), self.vocab_size))
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        self.multipliers = np.stack(
            [
                np.random.default_rng(self.seed + 10007 * layer).integers(
                    0, bound, size=self.max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in self.layer_ids
            ]
        )

        return True


def head_range(sizes, tp, rank):
    if not 1 <= tp <= len(sizes) or not 0 <= rank < tp:
        raise ValueError("Engram requires at least one complete hash head per TP rank")
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


class FlashNgramHashState(NgramHashState):
    def __init__(self, config, *, dummy=False):
        nn.Module.__init__(self)
        layout = FlashEngramLayout(config)
        self.layout: FlashEngramLayout = layout  # type: ignore[assignment]
        self.lookback_depth = self.layout.max_ngram_size - 1
        self.block_size = 1
        self.use_slot_cache = False
        self._cache = None
        self.pad_id = 0
        self.visual_token_ids: tuple[int, ...] = ()
        self.register_buffer(
            "token_map",
            torch.zeros(config.vocab_size, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.zeros(
                len(self.layout.layer_ids),
                self.layout.max_ngram_size,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        for name, values in (
            ("primes", self.layout.primes),
            ("offsets", self.layout.offsets),
        ):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.int64), persistent=False
            )
        if dummy:
            self.load_lookup(torch.arange(config.vocab_size, device="cpu"))

    def load_lookup(self, value):
        if self.layout.load_lookup(value):
            self.token_map.copy_(
                torch.as_tensor(
                    self.layout.token_map,
                    dtype=torch.int32,
                    device=self.token_map.device,
                )
            )
            self.multipliers.copy_(
                torch.as_tensor(self.layout.multipliers, device=self.multipliers.device)
            )
            self.pad_id = self.layout.token_map[self.layout.pad_token_id]

    @eager_break_during_capture
    def prepare_hashes_with_output(self, input_ids, lookback, layer_name, output):
        metadata = get_forward_context().attn_metadata
        result = self.prepare_hashes(
            input_ids,
            lookback,
            metadata[layer_name] if isinstance(metadata, dict) else None,
        )
        if os.environ.get('VEL')=='1' and not hasattr(self, "_shape_dbg"):
            self._shape_dbg = True
            import sys
            print(f"[SHAPE-DBG] result={tuple(result.shape)} output={tuple(output.shape)} "
                  f"nonneg_result={int((result >= 0).sum())}/{result.numel()}",
                  file=sys.stderr, flush=True)
        output.copy_(result)

    def prepare_hashes(self, input_ids, lookback, metadata):
        tokens = input_ids.numel()
        if metadata is None or tokens == 0:
            return input_ids.new_full(
                (len(self.layout.layer_ids), tokens, self.offsets.shape[1]),
                -1,
                dtype=torch.int32,
            )
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        starts = metadata.query_start_loc
        rows = torch.arange(tokens, device=input_ids.device)
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = (
            metadata.seq_lens[requests]
            - (starts[requests + 1] - starts[requests])
            + rows
            - starts[requests]
        ).long()
        dead = (input_ids < 0) | (input_ids >= self.token_map.numel())
        safe_ids = input_ids.clamp(0, self.token_map.numel() - 1)
        lookback_dead = lookback >= self.token_map.numel()
        safe_lookback = lookback.clamp(-1, self.token_map.numel() - 1)
        if os.environ.get('VEL')=='1' and not hasattr(self, "_hash_dbg"):
            self._hash_dbg = True
            import sys
            print(f"[HASH-DBG] tokens={tokens} starts={starts[:6].tolist()} "
                  f"num_actual={metadata.num_actual_tokens} lookback_shape={tuple(lookback.shape)} "
                  f"lookback_sample={lookback.flatten()[:6].tolist()} "
                  f"token_map_n={self.token_map.numel()} dead_n={int(dead.sum())} "
                  f"pad_id={self.layout.pad_token_id} visual={self.visual_token_ids}",
                  file=sys.stderr, flush=True)
        hashes = super().forward(
            safe_ids,
            logical,
            starts,
            dead,
            safe_lookback,
            lookback_dead,
        )
        if os.environ.get('VEL')=='1' and not hasattr(self, "_hash_dbg2"):
            self._hash_dbg2 = True
            import sys
            nz = int((hashes >= 0).sum())
            print(f"[HASH-DBG2] pre-mask hash>=0: {nz}/{hashes.numel()} sample={hashes.flatten()[:6].tolist()}",
                  file=sys.stderr, flush=True)
        # NO-DOUBLE-ADD: the base _hash_ids_kernel already applies primes+offsets
        # (hashed = rolling % prime + offset). Removing the redundant second apply.
        # offsets/primes retained only for debugging below.
        offsets = self.offsets.unsqueeze(0)
        primes = self.primes.flatten(1).unsqueeze(0)
        no_hash = dead | (rows >= metadata.num_actual_tokens) | (rows >= starts[-1])
        no_hash |= input_ids == self.layout.pad_token_id
        for token_id in self.visual_token_ids:
            no_hash |= input_ids == token_id
        hashes.masked_fill_(no_hash[:, None, None], -1)
        self._hash_dbg3_n = getattr(self, "_hash_dbg3_n", 6)
        if self._hash_dbg3_n < 6:
            self._hash_dbg3_n += 1
            import sys
            nz_after = int((hashes >= 0).sum())
            print(f"[HASH-DBG3] post-mask hash>=0: {nz_after}/{hashes.numel()} "
                  f"no_hash_true={int(no_hash.sum())}/{no_hash.numel()} "
                  f"rows_vs_actual: rows_max={int(rows.max())} num_actual={metadata.num_actual_tokens} "
                  f"starts_last={int(starts[-1])}",
                  file=sys.stderr, flush=True)
        return hashes.permute(1, 0, 2).contiguous()


# ---------------------------------------------------------------------------
# DISK-OFFLOAD BACKEND (added for SSD-resident Engram tables)
# ---------------------------------------------------------------------------
class _DiskRowStore:
    """Read-only row store over a checkpoint tensor file, using preadv.

    The table is never resident: only the rows a step actually needs are read.
    Rows for a (file, offset, row_bytes) tuple are contiguous, so a batch of row
    ids becomes a set of byte ranges we read in one pass.
    """

    def __init__(self, num_rows: int, row_bytes: int):
        self.num_rows = num_rows
        self.row_bytes = row_bytes
        self._path = None
        self._offset = 0
        self._fd = None

    def bind(self, path: str, offset: int) -> None:
        if self._fd is not None:
            os.close(self._fd)
        self._path = path
        self._offset = offset
        self._fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(self._fd, offset, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            pass

    def read_rows(self, row_ids, out: "torch.Tensor") -> None:
        """Copy `row_ids` into `out` (uint8 view, len(row_ids) x row_bytes)."""
        if self._fd is None:
            raise RuntimeError("DiskRowStore used before bind()")
        rb = self.row_bytes
        buf = memoryview(out.numpy().reshape(-1))
        import numpy as np

        ids = np.asarray(row_ids, dtype=np.int64)
        for i, rid in enumerate(ids):
            if rid < 0 or rid >= self.num_rows:
                buf[i * rb : (i + 1) * rb] = b"\x00" * rb
                continue
            off = self._offset + int(rid) * rb
            n = os.preadv(self._fd, [buf[i * rb : (i + 1) * rb]], off)
            if n != rb:
                raise RuntimeError(
                    f"short read at row {rid}: {n}/{rb} bytes from {self._path}"
                )

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class FlashEmbedding(nn.Module):
    def __init__(self, sizes, dim, dtype, cpu_offload, disk_offload=False):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(
            sizes, self.tp_size, get_tensor_model_parallel_rank()
        )
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.num_embeddings = sum(sizes)
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.cpu_offload = cpu_offload
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Flash Engram CPU offload requires UVA")
        rows = self.vocab_end - self.vocab_start
        storage = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.embedding = nn.Module()
        self.embedding.weight = nn.Parameter(
            torch.empty(
                (0,) if dtype is None else (rows, dim),
                dtype=dtype or torch.bfloat16,
                **storage,
            ),
            requires_grad=False,
        )
        self.fp8 = dtype == torch.float8_e4m3fn

        self.embedding.register_parameter("weight_scale_inv", None)
        if self.fp8:
            self.embedding.weight_scale_inv = nn.Parameter(
                torch.empty(rows, dtype=torch.bfloat16, **storage), requires_grad=False
            )
        for parameter in self.embedding.parameters():
            set_weight_attrs(parameter, {"weight_loader": self._load_shard})
        # --- DISK-OFFLOAD state (set true by the disk backend below)
        self.disk_offload = disk_offload
        self._row_store = None
        self._scale_store = None
        self._staging = None
        self._staging_rows = 0
        self._uva_weight = None
        self._uva_scale = None
        logger.info(
            "Flash Engram TP table: %d rows x %d, dtype=%s, cpu_offload=%s",
            rows,
            dim,
            dtype,
            cpu_offload,
        )

    def load_weight(self, leaf, value):
        if leaf == "weight_scale_inv" and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                self.embedding.weight.new_empty(0, dtype=torch.bfloat16),
                requires_grad=False,
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._load_shard(getattr(self.embedding, leaf), value)

    def _load_shard(self, parameter, value):
        is_weight = parameter is self.embedding.weight
        expected = (
            (self.num_embeddings, self.dim) if is_weight else (self.num_embeddings,)
        )
        dtypes = (
            (torch.bfloat16, torch.float8_e4m3fn) if is_weight else (torch.bfloat16,)
        )
        if tuple(value.shape) != expected or value.dtype not in dtypes:
            raise ValueError(
                f"Invalid Engram table: {tuple(value.shape)}/{value.dtype}"
            )
        if parameter.numel() and parameter.dtype != value.dtype:
            raise ValueError("Engram table dtype changed during weight reload")
        if is_weight:
            if (
                value.dtype != torch.float8_e4m3fn
                and self.embedding.weight_scale_inv is not None
            ):
                raise ValueError("BF16 Engram table must not have FP8 row scales")
        elif self.embedding.weight.numel() and not self.fp8:
            raise ValueError("BF16 Engram table must not have FP8 row scales")
        if self.disk_offload:
            # Disk backend: no resident table. Remember which checkpoint tensor this
            # rank's rows come from; the row store reads them on demand.
            base = getattr(self, "_disk_tensor_name", None)
            if base is None:
                raise RuntimeError(
                    "Flash Engram disk_offload: tensor name was not supplied by the loader"
                )
            if is_weight:
                self._tensor_name = base
            else:
                self._scale_name = base
            return
        if parameter.numel() == 0:
            parameter.data = torch.empty(
                (self.vocab_end - self.vocab_start, *expected[1:]),
                dtype=value.dtype,
                device=parameter.device,
                pin_memory=self.cpu_offload,
            )
        default_weight_loader(parameter, value[self.vocab_start : self.vocab_end])
        if is_weight:
            self.fp8 = value.dtype == torch.float8_e4m3fn
        if is_weight and self.fp8 and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                parameter.new_empty(0, dtype=torch.bfloat16), requires_grad=False
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._uva_weight = None
        self._uva_scale = None

    def _lookup(self, parameter, indices, out, *, scale=False):
        view_name = "_uva_scale" if scale else "_uva_weight"
        view = getattr(self, view_name)
        if view is None:
            view = (
                get_accelerator_view_from_cpu_tensor(parameter)
                if self.cpu_offload
                else parameter
            )
            setattr(self, view_name, view)
        row_bytes = parameter.element_size() * (1 if scale else self.dim)
        _lookup_ple_embedding_from_pinned_kernel[(indices.numel(),)](
            view.view(torch.uint8),
            indices,
            out.view(torch.uint8),
            row_bytes,
            self.vocab_start,
            self.vocab_end,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )

    def _bind_disk_store(self, parameter, *, scale=False):
        """Resolve the checkpoint byte range for this rank and open the row store."""
        import json
        import struct

        cfg_dir = getattr(self, "_pack_dir", None)
        if cfg_dir is None:
            raise RuntimeError(
                "Flash Engram disk_offload needs the pack directory; "
                "set FlashEmbedding._pack_dir before first use"
            )
        store = getattr(self, "_scale_store" if scale else "_row_store", None)
        row_bytes = parameter.element_size() * (1 if scale else self.dim)
        if store is not None:
            return store
        # Find the tensor's shard + data offset via the safetensors index.
        idx_path = os.path.join(cfg_dir, "model.safetensors.index.json")
        tname = getattr(self, "_tensor_name", None)
        if tname is None:
            raise RuntimeError("Flash Engram disk_offload: tensor name unknown")
        want = tname if not scale else tname.replace("weight", "weight_scale_inv")
        with open(idx_path) as f:
            wmap = json.load(f)["weight_map"]
        shard = wmap[want]
        shard_path = os.path.join(cfg_dir, shard)
        with open(shard_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n).decode())
        meta = hdr[want]
        base = 8 + n
        start = meta["data_offsets"][0]
        rows_total = meta["shape"][0]
        # our TP slice starts after all preceding heads' vocab
        row_off = start + self.vocab_start * row_bytes
        store = _DiskRowStore(rows_total, row_bytes)
        store.bind(shard_path, row_off)
        if scale:
            self._scale_store = store
        else:
            self._row_store = store
        logger.info(
            "Flash Engram disk store bound: %s rows=%d row_bytes=%d off=%d",
            shard, rows_total, row_bytes, row_off,
        )
        return store

    def _lookup_disk(self, parameter, indices, out, *, scale=False):
        """Gather only the needed rows from disk, then reuse the pinned-row kernel."""
        import numpy as np

        row_bytes = parameter.element_size() * (1 if scale else self.dim)
        store = self._bind_disk_store(parameter, scale=scale)
        if row_bytes != store.row_bytes:
            # scale rows differ in size from weight rows; use a second store slot
            store = getattr(self, "_scale_store", None)
            if store is None:
                raise RuntimeError("Flash Engram disk_offload: missing scale row store")

        # Pinned copy for CUDA-graph-safe D2H transfer
        _n = indices.numel()
        if not hasattr(self, "_idx_pin") or self._idx_pin.numel() < _n:
            self._idx_pin = torch.empty(max(_n, 8192), dtype=indices.dtype, pin_memory=True)
        _pin = self._idx_pin[:_n]
        # During CUDA graph capture, D2H + host I/O cannot be captured.
        # Use a separate non-capturing stream for the sync, or fall back to eager.
        _src = indices.reshape(-1)
        if torch.cuda.is_current_stream_capturing():
            # Disk I/O cannot be captured in CUDA graphs. Return zero rows during
            # capture; actual lookups happen in eager replay passes.
            out.zero_()
            return
        _pin.copy_(_src, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        flat_ids = _pin.numpy()
        uniq, inverse = np.unique(flat_ids, return_inverse=True)
        staged_rows = int(uniq.size)

        # grow the pinned staging buffer as needed
        if self._staging is None or self._staging_rows < staged_rows:
            self._staging_rows = max(staged_rows, 4096)
            self._staging = torch.empty(
                (self._staging_rows, row_bytes), dtype=torch.uint8, pin_memory=True
            )
        buf = self._staging[:staged_rows]
        store.read_rows(uniq, buf)

        mapped = torch.from_numpy(inverse.reshape(indices.shape)).to(
            indices.device, non_blocking=True
        )
        self._dbg_logged_n = getattr(self, "_dbg_logged_n", 0)
        if self._dbg_logged_n < 8:
            self._dbg_logged_n += 1
            import sys
            print(f"[ENGRAM-DBG] uniq={staged_rows} first_ids={uniq[:5].tolist()} "
                  f"row_bytes={row_bytes} offset={store._offset} path={store._path}", file=sys.stderr, flush=True)
            # Dump the first staged row raw bytes
            print(f"[ENGRAM-DBG] first row bytes={buf[0].numpy().tobytes()[:32].hex()}", file=sys.stderr, flush=True)
        _lookup_ple_embedding_from_pinned_kernel[(mapped.numel(),)](
            get_accelerator_view_from_cpu_tensor(buf).view(torch.uint8),
            mapped,
            out.view(torch.uint8),
            row_bytes,
            0,
            staged_rows,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )



    def forward(self, hashes):
        indices = hashes[:, self.head_start : self.head_end].long().contiguous()
        shape = (*indices.shape, self.dim)
        rows = torch.empty(
            shape, dtype=self.embedding.weight.dtype, device=hashes.device
        )
        if indices.numel() == 0:
            return rows.float()
        if self.disk_offload:
            self._lookup_disk(self.embedding.weight, indices, rows)
            if self.fp8:
                scales = torch.empty(
                    indices.shape, dtype=torch.bfloat16, device=hashes.device
                )
                self._lookup_disk(
                    self.embedding.weight_scale_inv, indices, scales, scale=True
                )
                return rows.float() * scales.float().unsqueeze(-1)
            return rows.float()
        self._lookup(self.embedding.weight, indices, rows)
        if self.fp8:
            scales = torch.empty(
                indices.shape, dtype=torch.bfloat16, device=hashes.device
            )
            self._lookup(self.embedding.weight_scale_inv, indices, scales, scale=True)
            return rows.float() * scales.float().unsqueeze(-1)
        return rows.float()


class FlashEngram(nn.Module):
    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        storage = vllm_config.engram_config
        if (
            storage.disk_offload
            or storage.embedding_across_dp
            or storage.dp_shared_memory
        ):
            if storage.disk_offload:
                # DISK-OFFLOAD-PATCH: keep the table in the checkpoint files; rows are
                # gathered through the page cache each step (mirrors DeepSeek-V4.1).
                disk = True
            else:
                raise ValueError(
                    "Flash Engram supports TP-local tables, not DP offload"
                )
        dtype = None
        if vllm_config.load_config.load_format == "dummy":
            dtype = (
                torch.float8_e4m3fn if config.engram_fp8_embedding else torch.bfloat16
            )
        self.multi_head_embedding = FlashEmbedding(
            sizes,
            layout.head_dim,
            dtype,
            storage.cpu_offload,
            bool(getattr(storage, "disk_offload", False)),
        )
        # Disk backend needs to know where the pack lives on disk.
        self.multi_head_embedding._pack_dir = vllm_config.model_config.model
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        local = (
            self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        )
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(layout.head_dim) for _ in range(local)]
        )
        self.value_proj = nn.Linear(local * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(local * layout.head_dim, self.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = self.multi_head_embedding.head_start * layout.head_dim
        self.stop = self.multi_head_embedding.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        for projection in [self.value_proj, *self.key_projs]:
            set_weight_attrs(
                projection.weight, {"weight_loader": self._load_projection}
            )
            set_weight_attrs(projection.bias, {"weight_loader": self._load_bias})

    def _load_projection(self, parameter, value):
        if value.shape != (self.hidden_size, self.total_width):
            raise ValueError("Flash Engram projection weight shape mismatch")
        default_weight_loader(parameter, value[:, self.start : self.stop])

    def _load_bias(self, parameter, value):
        default_weight_loader(
            parameter, value if self.rank == 0 else torch.zeros_like(value)
        )

    def forward(self, hidden, hashes):
        import os as _osd
        # ZERO-SKIP: skip the SSD row reads entirely in zero mode (quality knob),
        # returning a zero contribution keeps the add a no-op without preadv+syncs.
        if _osd.environ.get("VLLM_ENGRAM_ZERO") == "1":
            return hidden
        rows = self.multi_head_embedding(hashes)
        eps = torch.finfo(torch.float32).eps
        gamma = torch.stack(
            [norm.weight for norm in self.lookup_per_head_norms]
        ).float()
        local = self.multi_head_embedding
        local_hashes = hashes[:, local.head_start : local.head_end]
        x = rows.masked_fill(local_hashes.unsqueeze(-1) < 0, 0)
        x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma).to(
            hidden.dtype
        )
        x = x.flatten(1)
        gates = []
        for channel in range(self.hc_mult):
            key = self.key_projs[channel](x)
            if local.tp_size > 1:
                key = tensor_model_parallel_all_reduce(key)
            gate = (
                self.norm_key[channel](key)
                * self.norm_query[channel](hidden[:, channel])
            ).sum(-1)
            gate = gate * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = self.value_proj(x)
        if local.tp_size > 1:
            value = tensor_model_parallel_all_reduce(value)
        value = value.float()
        signal = (
            self.unified_pre_gamma.float()
            * value
            * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        )
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.float().unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(
            hidden.dtype
        )
        if _osd.environ.get("VLLM_ENGRAM_ZERO") == "1":
            return hidden
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)

