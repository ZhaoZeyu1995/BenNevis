"""
How `decode_k2.py` divides the work, and why the output order is explicit.

Tier 1: `read_scp`, `split_entries` and `in_scp_order` need only the
standard library. `shard_readers` needs kaldiio and torch and is not
exercised here -- what it adds over `split_entries` is one `load_mat`
call per entry.

**`output.scp` is sorted by duration, so a contiguous split is the worst
possible one.** It gives the last shard every one of the longest
utterances: on run 46100161743's 1,089 utterances the eight shards
finished across a five-minute spread, 08:16:59 to 08:22:03, with seven
idle at the end. Round-robin makes each shard a near-uniform sample of
the length distribution.

The previous code justified contiguous blocks by saying the recipe merges
against `ref.wrd.trn` **by position**, and that was simply wrong --
`read_trn` returns a mapping and `score` looks each utterance up by id.
What the order does decide is the bytes of `hyp.wrd.trn`, and CLAUDE.md
records that every `nj` from 1 to 8 gave byte-identical hypotheses. That
property used to hold by the accident of splitting contiguously; it now
holds because `in_scp_order` puts them back, which is why these tests
check it across every `nj` rather than at one.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "BenNevis" / "run"))

from decode_k2 import in_scp_order, read_scp, split_entries  # noqa: E402

#: An scp shaped like a real one: sorted by duration, so position
#: correlates with cost. The locations carry an offset, as Kaldi's do.
ENTRIES = [(f"utt{i:03d}", f"/ark/output.ark:{i * 1000}") for i in range(23)]


class TestReadScp:
    def test_it_reads_pairs_in_order(self, tmp_path):
        scp = tmp_path / "output.scp"
        scp.write_text(
            "".join(f"{utt} {where}\n" for utt, where in ENTRIES), encoding="utf-8"
        )
        assert read_scp(scp) == ENTRIES

    def test_a_malformed_line_is_skipped(self, tmp_path):
        """
        What `kaldiio` does with it too. A key with no location cannot be
        read from, and failing the whole decode over one would lose a run
        that is hours in.
        """
        scp = tmp_path / "output.scp"
        scp.write_text("utt000 /ark/a:0\nutt001\n\nutt002 /ark/a:2\n", encoding="utf-8")
        assert [utt for utt, _ in read_scp(scp)] == ["utt000", "utt002"]

    def test_a_location_keeps_its_own_spacing(self, tmp_path):
        scp = tmp_path / "output.scp"
        scp.write_text("utt000 /ark/with space.ark:12\n", encoding="utf-8")
        assert read_scp(scp) == [("utt000", "/ark/with space.ark:12")]


class TestTheSplit:
    @pytest.mark.parametrize("count", [1, 2, 3, 4, 8, 23])
    def test_every_utterance_appears_exactly_once(self, count):
        blocks = split_entries(ENTRIES, count)
        flattened = [entry for block in blocks for entry in block]
        assert sorted(flattened) == sorted(ENTRIES)
        assert len(flattened) == len(ENTRIES)

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
    def test_the_shards_are_balanced(self, count):
        """
        The whole point. With a contiguous split of 23 entries into 8,
        the blocks are 3,3,3,3,3,3,3,2 by *count* -- but the last holds
        the longest utterances, so it is the slowest by a wide margin.
        Round-robin balances the lengths, and the counts still differ by
        at most one.
        """
        sizes = [len(block) for block in split_entries(ENTRIES, count)]
        assert len(sizes) == count
        assert max(sizes) - min(sizes) <= 1

    def test_each_shard_samples_the_whole_range(self, count=4):
        """
        The property a contiguous split lacks: every shard must contain
        short and long utterances. Checked as the mean position, which
        for a uniform sample is near the middle for every shard.
        """
        middle = (len(ENTRIES) - 1) / 2
        positions = {utt: i for i, (utt, _) in enumerate(ENTRIES)}
        for block in split_entries(ENTRIES, count):
            mean = sum(positions[utt] for utt, _ in block) / len(block)
            assert abs(mean - middle) < 2.0, f"{mean} against {middle}"

    def test_a_contiguous_split_would_fail_that(self, count=4):
        """
        The counter-example, so the test above is known to discriminate.
        A test that passes for both splits would pin nothing.
        """
        middle = (len(ENTRIES) - 1) / 2
        positions = {utt: i for i, (utt, _) in enumerate(ENTRIES)}
        size = -(-len(ENTRIES) // count)
        contiguous = [ENTRIES[i : i + size] for i in range(0, len(ENTRIES), size)]
        means = [
            sum(positions[utt] for utt, _ in block) / len(block) for block in contiguous
        ]
        assert any(abs(mean - middle) >= 2.0 for mean in means)

    def test_more_shards_than_utterances_makes_no_empty_shard(self):
        """
        A shard yielding nothing still forks a process to do nothing.
        """
        blocks = split_entries(ENTRIES[:3], 8)
        assert len(blocks) == 3
        assert all(block for block in blocks)

    def test_an_empty_scp_gives_one_empty_shard(self):
        assert split_entries([], 8) == [[]]

    @pytest.mark.parametrize("count", [0, -1])
    def test_a_nonsense_count_is_clamped(self, count):
        assert len(split_entries(ENTRIES, count)) == 1


class TestTheOutputOrder:
    @staticmethod
    def decode(entries, count):
        """
        Simulate a sharded decode: each shard returns its own block, and
        the shards' results are concatenated in shard order, which is
        what `decode_shards` does.
        """
        hypotheses = []
        for block in split_entries(entries, count):
            hypotheses.extend((utt, [utt.upper()]) for utt, _ in block)
        return hypotheses

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
    def test_the_order_is_the_scps_for_every_nj(self, count):
        """
        CLAUDE.md records that every `nj` from 1 to 8 gave byte-identical
        hypotheses, which is what makes the flag safe. That used to hold
        by the accident of splitting contiguously; now it holds because
        the order is restored.
        """
        ordered = in_scp_order(self.decode(ENTRIES, count), ENTRIES)
        assert [utt for utt, _ in ordered] == [utt for utt, _ in ENTRIES]

    def test_every_nj_gives_the_same_list(self):
        """
        Stated as an equality between runs rather than against the scp,
        because that is the property the flag's safety rests on.
        """
        results = {
            count: in_scp_order(self.decode(ENTRIES, count), ENTRIES)
            for count in (1, 2, 3, 4, 8, 23)
        }
        first = results[1]
        for count, ordered in results.items():
            assert ordered == first, f"nj={count} differs"

    def test_an_utterance_not_in_the_scp_is_appended_not_dropped(self):
        """
        Losing an utterance silently is the failure this project keeps
        meeting -- and `score` fails loudly on a missing one, so an
        unexpected extra must survive to be seen.
        """
        hypotheses = [("ghost", ["X"]), *self.decode(ENTRIES, 4)]
        ordered = in_scp_order(hypotheses, ENTRIES)
        assert len(ordered) == len(ENTRIES) + 1
        assert ordered[-1][0] == "ghost"

    def test_several_unknown_utterances_keep_their_arrival_order(self):
        hypotheses = [("zz", ["Z"]), ("aa", ["A"])]
        ordered = in_scp_order(hypotheses, ENTRIES)
        assert [utt for utt, _ in ordered] == ["zz", "aa"]

    def test_a_missing_utterance_is_not_invented(self):
        """
        The other direction: reordering must not fill a gap. `score`
        refuses a hypothesis that is short, and that is the behaviour
        wanted -- a decoder that dropped utterances would otherwise score
        well on the ones it kept.
        """
        partial = self.decode(ENTRIES, 4)[:-3]
        ordered = in_scp_order(partial, ENTRIES)
        assert len(ordered) == len(partial)

    def test_the_words_travel_with_their_utterance(self):
        ordered = in_scp_order(self.decode(ENTRIES, 8), ENTRIES)
        for utt, words in ordered:
            assert words == [utt.upper()]
