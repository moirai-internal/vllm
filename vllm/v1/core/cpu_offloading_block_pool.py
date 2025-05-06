# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue, BlockHashType

logger = init_logger(__name__)


class CpuOffloadingBlockPool(BlockPool):
    """
    CpuOffloadingBlockPool that manages GPU and CPU offloading KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of gpu blocks in the pool.
        num_cpu_blocks: The number of cpu blocks in the pool.
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int):
        logger.info("enable cpu offloading block pool, num gpu blocks: %s, num cpu blocks: %s",
                    num_gpu_blocks, num_cpu_blocks)
        assert num_cpu_blocks > num_gpu_blocks, ("CpuOffloadingBlockPool requires the allocated "
                                                 "CPU memory capacity to be larger than GPU memory capacity.")
        super().__init__(num_gpu_blocks, True)

        # cpu offloading
        self.num_cpu_blocks = num_cpu_blocks
        self.cpu_blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_cpu_blocks)
        ]
        self.free_cpu_block_queue = FreeKVCacheBlockQueue(self.cpu_blocks)

        # {block_hash: {block ID: block}}
        self.cached_block_hash_to_cpu_block: dict[BlockHashType, dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # swap GPU block to CPU block, GPU block ID -> CPU block ID
        self.step_d2h_swap_map: dict[int, int] = {}
        # swap CPU block to GPU block, CPU block ID -> GPU block ID
        self.step_h2d_swap_map: dict[int, int] = {}

        self.swap_in_count: int = 0
        self.swap_out_count: int = 0

        self.cpu_evict_count: int = 0

    def get_new_blocks(self, num_blocks: int,
                       computed_cpu_blocks: Optional[list[KVCacheBlock]] = None,
                       cpu_offloading: bool = True,) -> list[KVCacheBlock]:
        num_cpu_blocks = num_blocks - len(computed_cpu_blocks) if computed_cpu_blocks else 0
        assert num_cpu_blocks >= 0
        if num_blocks > self.get_num_free_blocks() \
                or num_cpu_blocks > self.get_num_free_cpu_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the the gpu and cpu pool")

        ret: list[KVCacheBlock] = []
        idx = 0
        computed_cpu_blocks_len = len(computed_cpu_blocks) if computed_cpu_blocks else 0
        while idx < num_blocks:
            # First allocate GPU blocks.
            curr_block = self.free_block_queue.popleft()
            assert curr_block.ref_cnt == 0
            self._maybe_evict_cached_block(curr_block)
            curr_block.cpu_offloading_block = None
            curr_block.incr_ref()

            # Then allocate CPU blocks.
            if cpu_offloading:
                if idx < computed_cpu_blocks_len:
                    # swap in
                    cpu_block = computed_cpu_blocks[idx]
                    if cpu_block.ref_cnt == 0:
                        self.free_cpu_block_queue.remove(cpu_block)
                    self.step_h2d_swap_map[cpu_block.block_id] = curr_block.block_id
                    self.swap_in_count += 1
                else:
                    cpu_block = self.free_cpu_block_queue.popleft()
                    assert cpu_block.ref_cnt == 0
                    self._maybe_evict_cached_cpu_block(cpu_block)
                cpu_block.incr_ref()
                curr_block.cpu_offloading_block = cpu_block

            ret.append(curr_block)
            idx += 1

        return ret

    def cache_full_cpu_blocks(self, blocks: list[KVCacheBlock],
                              num_cached_cpu_blocks: int,
                              num_full_blocks: int,
                              ) -> None:
        assert num_cached_cpu_blocks <= num_full_blocks
        if num_cached_cpu_blocks == num_full_blocks:
            return
        # blocks is GPU blocks, use block.cpu_offloading_block to update CPU cache
        new_full_cpu_blocks = blocks[num_cached_cpu_blocks:num_full_blocks]
        for block in new_full_cpu_blocks:
            assert block.block_hash is not None
            if block.cpu_offloading_block is not None:
                assert block.cpu_offloading_block.block_hash is None
                block.cpu_offloading_block.block_hash = block.block_hash
                self.cached_block_hash_to_cpu_block[block.block_hash][block.cpu_offloading_block.block_id] = block.cpu_offloading_block

                # swap out
                self.step_d2h_swap_map[block.block_id] = block.cpu_offloading_block.block_id
                self.swap_out_count += 1
            else:
                break

    def _maybe_evict_cached_cpu_block(self, cpu_block: KVCacheBlock) -> bool:
        """
        If a cpu block is cached in `cached_block_hash_to_cpu_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            cpu_block: The cpu block to evict.

        Returns:
            True if the cpu block is evicted, False otherwise.
        """
        block_hash = cpu_block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_cpu_block:
            self.cpu_evict_count += 1
            cpu_block.reset_hash()
            del self.cached_block_hash_to_cpu_block[block_hash][cpu_block.block_id]

            if len(self.cached_block_hash_to_cpu_block[block_hash]) == 0:
                del self.cached_block_hash_to_cpu_block[block_hash]

            return True
        return False

    def get_cached_cpu_block(
            self, block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached cpu block by the block hash, or None if cache miss.

        Args:
            block_hash: The hash value of the block.

        Returns:
            The cached cpu block if it exists, or None.
        """
        cached_cpu_blocks = self.cached_block_hash_to_cpu_block.get(block_hash)
        if not cached_cpu_blocks:
            return None
        first_cpu_block_id = next(iter(cached_cpu_blocks))
        return cached_cpu_blocks[first_cpu_block_id]

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            if block.cpu_offloading_block is not None:
                if block.cpu_offloading_block.ref_cnt == 0:
                    self.free_cpu_block_queue.remove(block.cpu_offloading_block)
                block.cpu_offloading_block.incr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.incr_ref()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        for block in ordered_blocks:
            if block.cpu_offloading_block is not None:
                block.cpu_offloading_block.decr_ref()
                if block.cpu_offloading_block.ref_cnt == 0:
                    self.free_cpu_block_queue.append(block.cpu_offloading_block)
            block.decr_ref()
            if block.ref_cnt == 0:
                self.free_block_queue.append(block)

    def get_num_free_cpu_blocks(self) -> int:
        """Get the number of free blocks in the cpu pool.

        Returns:
            The number of free cpu blocks.
        """
        return self.free_cpu_block_queue.num_free_blocks

    def get_cpu_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return 1.0 - (self.get_num_free_cpu_blocks() / self.num_cpu_blocks)

    def get_cpu_evict_count(self) -> int:
        return self.cpu_evict_count

    def clear_step_d2h_swap_map(self) -> None:
        """
        Clear swap out map
        """
        self.step_d2h_swap_map.clear()

    def clear_step_h2d_swap_map(self) -> None:
        """
        Clear swap in map
        """
        self.step_h2d_swap_map.clear()
