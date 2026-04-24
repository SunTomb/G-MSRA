"""
Memory Compactor: LLM-based memory consolidation (v2).

Replaces LoRA-based SemanticDistiller with a non-parametric approach:
instead of distilling memories into model weights, we compact the
memory store itself by clustering similar entries and merging them
via LLM summarization.

Key advantage: model weights remain untouched → no catastrophic forgetting.

Flow:
  1. Find clusters of semantically similar memories (cosine > threshold)
  2. For each cluster, use LLM to generate a concise summary
  3. DELETE original entries, ADD the summary as a single new entry
  4. Result: fewer, higher-quality memories → better retrieval
"""

from __future__ import annotations
from typing import Optional
import time

import numpy as np
from loguru import logger

from gmsra.config import CompactionConfig
from gmsra.memory.store import MemoryStore
from gmsra.memory.entry import MemoryEntry


class MemoryCompactor:
    """Non-parametric memory compaction via LLM summarization.

    Unlike SemanticDistiller (v1), this module:
    - Does NOT modify model weights
    - Operates purely on the external memory store
    - Uses LLM to merge clusters of similar memories into concise summaries
    """

    def __init__(self, config: Optional[CompactionConfig] = None):
        self.config = config or CompactionConfig()
        self.compaction_count = 0
        self.stats_history: list[dict] = []

    def find_clusters(self, store: MemoryStore) -> list[list[str]]:
        """Find clusters of semantically similar memories.

        Uses greedy agglomerative clustering: for each entry, find all
        entries with cosine similarity > threshold, group them.

        Returns:
            List of clusters, each cluster is a list of entry IDs.
        """
        if store.size() < self.config.trigger_memory_threshold:
            return []

        entries = list(store.entries.values())
        if len(entries) < self.config.min_cluster_size:
            return []

        # Collect embeddings
        store._init_encoder()
        embeddings = []
        entry_ids = []
        for entry in entries:
            if entry.embedding:
                embeddings.append(np.array(entry.embedding))
                entry_ids.append(entry.id)
            else:
                # Re-encode if embedding is missing
                emb = store._encode(entry.content)
                embeddings.append(emb)
                entry_ids.append(entry.id)

        if len(embeddings) < self.config.min_cluster_size:
            return []

        emb_matrix = np.stack(embeddings)
        # Normalize for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_normed = emb_matrix / norms

        # Compute pairwise cosine similarity
        sim_matrix = emb_normed @ emb_normed.T

        # Greedy clustering
        used = set()
        clusters = []

        for i in range(len(entry_ids)):
            if entry_ids[i] in used:
                continue

            # Find all entries similar to i
            cluster = [entry_ids[i]]
            used.add(entry_ids[i])

            for j in range(i + 1, len(entry_ids)):
                if entry_ids[j] in used:
                    continue
                if sim_matrix[i, j] >= self.config.similarity_threshold:
                    cluster.append(entry_ids[j])
                    used.add(entry_ids[j])
                    if len(cluster) >= self.config.max_cluster_size:
                        break

            if len(cluster) >= self.config.min_cluster_size:
                clusters.append(cluster)

        return clusters

    def compact_cluster(
        self,
        cluster_ids: list[str],
        store: MemoryStore,
        model=None,
        tokenizer=None,
    ) -> Optional[str]:
        """Merge a cluster of similar memories into a single concise summary.

        Args:
            cluster_ids: IDs of memories to merge.
            store: Memory store containing the entries.
            model: LLM for summarization.
            tokenizer: Tokenizer for the LLM.

        Returns:
            The summary text, or None if compaction failed.
        """
        # Collect content from cluster
        contents = []
        for eid in cluster_ids:
            if eid in store.entries:
                contents.append(store.entries[eid].content)

        if len(contents) < self.config.min_cluster_size:
            return None

        # Build compaction prompt
        memory_list = "\n".join(
            f"  {i+1}. {c}" for i, c in enumerate(contents)
        )
        prompt = (
            "You are a memory manager. The following memory entries are about "
            "the same or very similar topics. Merge them into a SINGLE concise "
            "factual statement that preserves all important information.\n\n"
            "If there are contradictory facts (e.g., the user moved from city A "
            "to city B), keep only the LATEST information.\n\n"
            f"Memory entries to merge:\n{memory_list}\n\n"
            "Merged summary (one concise sentence):"
        )

        if model is not None and tokenizer is not None:
            from gmsra.utils import generate_text
            summary = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=self.config.summary_max_tokens,
                temperature=0.3,
            )
            return summary.strip()
        else:
            # Fallback: concatenate with dedup (no LLM available)
            unique = list(dict.fromkeys(contents))  # preserve order, dedup
            return " | ".join(unique[:3])

    def run(
        self,
        store: MemoryStore,
        model=None,
        tokenizer=None,
    ) -> dict:
        """Run one round of memory compaction.

        Returns:
            Stats dict with compaction results.
        """
        start_time = time.time()
        initial_size = store.size()

        clusters = self.find_clusters(store)
        if not clusters:
            stats = {
                "skipped": True,
                "reason": "no clusters found" if initial_size >= self.config.trigger_memory_threshold
                          else f"store too small ({initial_size} < {self.config.trigger_memory_threshold})",
                "initial_size": initial_size,
                "final_size": initial_size,
            }
            logger.info(f"Compaction skipped: {stats['reason']}")
            return stats

        merged_count = 0
        deleted_count = 0
        summaries = []

        for cluster_ids in clusters:
            summary = self.compact_cluster(cluster_ids, store, model, tokenizer)
            if summary is None:
                continue

            # DELETE original entries
            for eid in cluster_ids:
                if store.delete(eid):
                    deleted_count += 1

            # ADD the merged summary
            store.add(
                content=summary,
                env_reward=0.5,  # Neutral reward for compacted memories
                source=f"compaction_{self.compaction_count}",
            )
            merged_count += 1
            summaries.append({
                "cluster_size": len(cluster_ids),
                "summary": summary[:100],
            })

        final_size = store.size()
        elapsed = time.time() - start_time
        self.compaction_count += 1

        stats = {
            "skipped": False,
            "compaction_id": self.compaction_count,
            "initial_size": initial_size,
            "final_size": final_size,
            "reduction": initial_size - final_size,
            "clusters_found": len(clusters),
            "clusters_merged": merged_count,
            "entries_deleted": deleted_count,
            "entries_added": merged_count,
            "elapsed_seconds": round(elapsed, 2),
            "summaries": summaries[:5],  # Keep first 5 for logging
        }
        self.stats_history.append(stats)

        logger.info(
            f"Compaction #{self.compaction_count}: "
            f"{initial_size} → {final_size} memories "
            f"({len(clusters)} clusters, {deleted_count} deleted, "
            f"{merged_count} summaries added, {elapsed:.1f}s)"
        )

        return stats
