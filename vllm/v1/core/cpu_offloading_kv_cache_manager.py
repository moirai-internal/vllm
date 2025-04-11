# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.cpu_offloading_block_pool import CpuOffloadingBlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, hash_request_tokens
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


# TODO(idellzheng): support sliding window attention
class CpuOffloadingKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        offloading_blocks_threshold: int = 1,
    ) -> None:
        logger.info(f"use CpuOffloadingKVCacheManager, offloading threshold: {offloading_blocks_threshold}.")
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            caching_hash_algo=caching_hash_algo,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events
        )
        assert enable_caching
        self.offloading_blocks_threshold = offloading_blocks_threshold
        self.num_cpu_blocks = kv_cache_config.num_cpu_blocks
        self.block_pool = CpuOffloadingBlockPool(self.num_gpu_blocks, self.num_cpu_blocks)
        self.cpu_prefix_cache_stats: PrefixCacheStats = PrefixCacheStats()

    def get_computed_blocks(
            self, request: Request
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock], int]:
        """Get the computed (cached) GPU and CPU blocks for the request.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of gpu blocks that are computed for the request.
                - A list of cpu blocks that are computed for the request.
                - The number of computed tokens including gpu and cpu blocks.
        """

        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.caching_hash_fn,
                                               self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        self.prefix_cache_stats.requests += 1
        if request.sampling_params.prompt_logprobs is None:
            # Check for cache hits
            computed_blocks = []
            computed_cpu_blocks = []
            for block_hash in block_hashes:
                # NOTE: Order matters here, since a block might reside both in gpu
                # and cpu cache, we check cpu cache only if gpu cache is a miss.
                if cached_block := self.block_pool.get_cached_block(block_hash):
                    computed_blocks.append(cached_block)
                elif cached_cpu_block := self.block_pool.get_cached_cpu_block(block_hash):
                    computed_cpu_blocks.append(cached_cpu_block)
                else:
                    break
            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)
            self.cpu_prefix_cache_stats.queries += (len(block_hashes) - len(computed_blocks))
            self.cpu_prefix_cache_stats.hits += len(computed_cpu_blocks)

            num_computed_tokens = (len(computed_blocks) +
                                   len(computed_cpu_blocks)) * self.block_size
            return computed_blocks, computed_cpu_blocks, num_computed_tokens
        else:
            # Skip cache hits for prompt logprobs
            return [], [], 0

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        num_lookahead_tokens: int = 0,
        new_computed_cpu_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            new_computed_cpu_blocks: A list of new computed cpu blocks just
                hitting the prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed gpu:cpu > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []
        new_computed_cpu_blocks = new_computed_cpu_blocks or []

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        total_tokens = (
            request.num_computed_tokens +
            (len(new_computed_blocks) + len(new_computed_cpu_blocks)) *
            self.block_size + num_tokens + num_lookahead_tokens)
        num_required_blocks = cdiv(total_tokens, self.block_size)
        req_blocks = self.req_to_blocks[request.request_id]
        # New blocks needed, including GPU blocks for CPU cache hits
        # that need to be swapped in.
        num_new_blocks = (num_required_blocks - len(req_blocks) - len(new_computed_blocks))

        # For prefill, if num_new_blocks is greater than 1, it triggers CPU offloading
        # For decode, num_new_blocks is always less than 1, so it will not trigger CPU offloading
        cpu_offloading = num_new_blocks > self.offloading_blocks_threshold

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(1 for blk in new_computed_blocks
                                            if blk.ref_cnt == 0)
        if (num_new_blocks > self.block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks
            logger.warning(f"allocate gpu blocks failed for {request.request_id}, "
                           f"need new blocks {num_new_blocks}, "
                           f"free gpu blocks {self.block_pool.get_num_free_blocks()} "
                           f"which includes cached blocks {num_evictable_computed_blocks}")
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        self.block_pool.touch(new_computed_blocks)

        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks
        if num_new_blocks <= 0:
            new_blocks = []
        else:
            num_new_blocks = min(
                num_new_blocks + (0 if cpu_offloading else self.num_preallocate_blocks),
                self.block_pool.get_num_free_blocks(),
                self.max_num_blocks_per_req - len(req_blocks),
            )
            assert num_new_blocks > 0

            new_blocks = self.block_pool.get_new_blocks(num_new_blocks,
                                                        new_computed_cpu_blocks,
                                                        cpu_offloading,)
            req_blocks.extend(new_blocks)

        num_cached_blocks = self.num_cached_block.get(request.request_id,
                                                      len(new_computed_blocks))
        num_full_blocks_after_append = (
            total_tokens - len(request.spec_token_ids)) // self.block_size

        # cache GPU blocks
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
            hash_fn=self.caching_hash_fn,
        )
        # cache CPU blocks
        self.block_pool.cache_full_cpu_blocks(
            blocks=req_blocks,
            num_cached_cpu_blocks=num_cached_blocks + len(new_computed_cpu_blocks),
            num_full_blocks=num_full_blocks_after_append,
        )

        self.num_cached_block[request.request_id] = num_full_blocks_after_append
        return new_blocks

    def get_d2h_swap_map(self) -> dict[int, int]:
        """Get d2h (GPU -> CPU) swap map

        NOTE: use copy to prevent clearing in `end_schedule_step`
        """
        return self.block_pool.step_d2h_swap_map.copy()

    def get_h2d_swap_map(self) -> dict[int, int]:
        """Get h2d (CPU -> GPU) swap map

        NOTE: use copy to prevent clearing in `end_schedule_step`
        """
        return self.block_pool.step_h2d_swap_map.copy()

    def clear_step_d2h_swap_map(self) -> None:
        """
        Clear swap out map if enable cpu offloading
        """
        self.block_pool.clear_step_d2h_swap_map()

    def clear_step_h2d_swap_map(self) -> None:
        """
        Clear swap in map if enable cpu offloading
        """
        self.block_pool.clear_step_h2d_swap_map()

    def make_cpu_prefix_cache_stats(self) -> PrefixCacheStats:
        """Get (and reset) the cpu prefix cache stats.

        Returns:
            The current cpu prefix caching stats.
        """
        stats = self.cpu_prefix_cache_stats
        self.cpu_prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_swap_in_count(self) -> int:
        """Get the swap in count(CPU -> GPU).

        Returns:
            The total number of swap in.
        """
        return self.block_pool.swap_in_count

    def get_swap_out_count(self) -> int:
        """Get the swap out count(GPU -> CPU).

        Returns:
            The total number of swap out.
        """
        return self.block_pool.swap_out_count

    @property
    def cpu_usage(self) -> float:
        """Get the CPU KV cache usage.

        Returns:
            The CPU KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_cpu_usage()

    def get_gpu_cpu_evict_count(self) -> tuple[int, int]:
        return self.block_pool.get_evict_count(), self.block_pool.get_cpu_evict_count()
