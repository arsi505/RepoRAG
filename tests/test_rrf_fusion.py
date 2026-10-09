from __future__ import annotations

import math
import unittest

from reporag.fusion.rrf import RankedItem, RRFFusionError, reciprocal_rank_fusion


def item(chunk_id: str, rank: int, score: float = 1.0) -> RankedItem:
    return RankedItem(chunk_id, rank, score)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_result_in_both_rankings_receives_two_contributions(self) -> None:
        result = reciprocal_rank_fusion([item("both", 2)], [item("both", 3)])[0]
        self.assertGreater(result.vector_contribution, 0.0)
        self.assertGreater(result.bm25_contribution, 0.0)
        self.assertAlmostEqual(result.rrf_score, 1 / 62 + 1 / 63)

    def test_vector_only_result_receives_one_contribution(self) -> None:
        result = reciprocal_rank_fusion([item("vector", 1)], [])[0]
        self.assertEqual(result.vector_contribution, 1 / 61)
        self.assertEqual(result.bm25_contribution, 0.0)
        self.assertIsNone(result.bm25_rank)

    def test_bm25_only_result_receives_one_contribution(self) -> None:
        result = reciprocal_rank_fusion([], [item("bm25", 1, 9.0)])[0]
        self.assertEqual(result.vector_contribution, 0.0)
        self.assertEqual(result.bm25_contribution, 1 / 61)
        self.assertIsNone(result.vector_rank)

    def test_formula_uses_one_based_rank(self) -> None:
        result = reciprocal_rank_fusion([item("one", 1)], [])[0]
        self.assertAlmostEqual(result.rrf_score, 1 / (60 + 1))

    def test_default_k_is_frozen_at_60(self) -> None:
        self.assertAlmostEqual(
            reciprocal_rank_fusion([item("one", 1)], [])[0].rrf_score, 1 / 61
        )

    def test_changing_k_changes_score(self) -> None:
        sixty = reciprocal_rank_fusion([item("one", 1)], [], k=60)[0]
        ten = reciprocal_rank_fusion([item("one", 1)], [], k=10)[0]
        self.assertNotEqual(sixty.rrf_score, ten.rrf_score)

    def test_ranking_is_deterministic(self) -> None:
        vector = [item("a", 2), item("b", 1)]
        bm25 = [item("a", 1), item("c", 2)]
        self.assertEqual(
            reciprocal_rank_fusion(vector, bm25),
            reciprocal_rank_fusion(vector, bm25),
        )

    def test_ties_use_chunk_id_not_raw_score(self) -> None:
        results = reciprocal_rank_fusion(
            [item("z-last", 1, 999.0)], [item("a-first", 1, 0.001)]
        )
        self.assertEqual([result.chunk_id for result in results], ["a-first", "z-last"])

    def test_duplicate_chunk_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(RRFFusionError, "Duplicate chunk_id"):
            reciprocal_rank_fusion([item("same", 1), item("same", 2)], [])

    def test_duplicate_ranks_are_rejected(self) -> None:
        with self.assertRaisesRegex(RRFFusionError, "Duplicate rank"):
            reciprocal_rank_fusion([item("a", 1), item("b", 1)], [])

    def test_invalid_ranks_are_rejected(self) -> None:
        for rank in (0, -1, 1.5, True):
            with self.subTest(rank=rank), self.assertRaisesRegex(RRFFusionError, "1-based"):
                reciprocal_rank_fusion([item("bad", rank)], [])

    def test_invalid_k_is_rejected(self) -> None:
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaisesRegex(RRFFusionError, "positive integer"):
                reciprocal_rank_fusion([], [], k=k)

    def test_both_empty_rankings_return_empty_result(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([], []), ())

    def test_each_one_empty_ranker_is_supported(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([item("v", 1)], [])[0].chunk_id, "v")
        self.assertEqual(reciprocal_rank_fusion([], [item("b", 1)])[0].chunk_id, "b")

    def test_non_finite_raw_scores_are_rejected(self) -> None:
        for score in (math.nan, math.inf, -math.inf):
            with self.subTest(score=score), self.assertRaisesRegex(RRFFusionError, "finite"):
                reciprocal_rank_fusion([item("bad", 1, score)], [])

    def test_raw_scores_do_not_change_rrf_order(self) -> None:
        low_high = reciprocal_rank_fusion(
            [item("first", 1, -1000.0), item("second", 2, 1000.0)], []
        )
        high_low = reciprocal_rank_fusion(
            [item("first", 1, 1000.0), item("second", 2, -1000.0)], []
        )
        self.assertEqual(
            [result.chunk_id for result in low_high],
            [result.chunk_id for result in high_low],
        )

    def test_final_top_k_is_applied_after_full_fusion(self) -> None:
        results = reciprocal_rank_fusion(
            [item("vector-first", 1), item("both", 2)],
            [item("bm25-first", 1), item("both", 2)],
            top_k=1,
        )
        self.assertEqual(results[0].chunk_id, "both")

    def test_invalid_top_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(RRFFusionError, "top_k"):
            reciprocal_rank_fusion([], [], top_k=0)


if __name__ == "__main__":
    unittest.main()
