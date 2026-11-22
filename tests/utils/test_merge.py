"""
Tests for merging the per-rank files a distributed prediction leaves.

Tier 1: nothing but the standard library. These replace
`cat <glob> | awk '!seen[$N]++'`, which had no tests because it lived in an
entry-point script -- reachable from no tier but 3, which is how the
regression in #5 got in.

The deduplication is pinned exactly as `awk` did it, because the WER
depends on it: a duplicated utterance would be scored twice and a dropped
one would make the hypothesis shorter than the reference.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from BenNevis.utils.merge import FIRST_FIELD, WHOLE_LINE, merge_unique, rank_sorted


def write(tmp_path, name, *lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestRankSorted:
    def test_numbers_sort_as_numbers(self, tmp_path):
        """
        The bug in the shell version. A glob is lexicographic, so eleven
        ranks merged as 1, 10, 11, 2, 3 ... Nothing downstream depended on
        the order, because both files are keyed by utterance, but
        reproducing it in Python would be reproducing a mistake.
        """
        names = [f"output.{index}.scp" for index in (1, 2, 10, 11, 3)]
        paths = [write(tmp_path, name, "x") for name in names]
        assert [p.name for p in rank_sorted(paths)] == [
            "output.1.scp",
            "output.2.scp",
            "output.3.scp",
            "output.10.scp",
            "output.11.scp",
        ]

    def test_a_shell_glob_would_not(self, tmp_path):
        """
        States the thing being fixed, so the test says why it exists.
        """
        names = sorted(f"output.{index}.scp" for index in (1, 2, 10))
        assert names == ["output.1.scp", "output.10.scp", "output.2.scp"]

    def test_single_digit_ranks_are_unaffected(self, tmp_path):
        paths = [write(tmp_path, f"a.{i}.scp", "x") for i in (3, 1, 2)]
        assert [p.name for p in rank_sorted(paths)] == [
            "a.1.scp",
            "a.2.scp",
            "a.3.scp",
        ]

    def test_it_is_stable_for_names_without_numbers(self, tmp_path):
        paths = [write(tmp_path, n, "x") for n in ("b.scp", "a.scp")]
        assert [p.name for p in rank_sorted(paths)] == ["a.scp", "b.scp"]


class TestMergeUnique:
    def test_it_concatenates_in_order(self, tmp_path):
        first = write(tmp_path, "a", "one", "two")
        second = write(tmp_path, "b", "three")
        assert merge_unique([first, second]) == "one\ntwo\nthree\n"

    def test_a_repeated_line_is_dropped(self, tmp_path):
        first = write(tmp_path, "a", "one", "two")
        second = write(tmp_path, "b", "two", "three")
        assert merge_unique([first, second]) == "one\ntwo\nthree\n"

    def test_the_first_occurrence_wins(self, tmp_path):
        """
        As `awk '!seen[...]++'` does. Which copy survives matters for
        `output.scp`, whose later fields are an archive offset that differs
        between ranks -- the surviving one has to point at a file that
        exists.
        """
        first = write(tmp_path, "a", "utt1 first")
        second = write(tmp_path, "b", "utt1 second")
        assert merge_unique([first, second], key=FIRST_FIELD) == "utt1 first\n"

    def test_the_first_field_key_ignores_the_rest(self, tmp_path):
        first = write(tmp_path, "a", "utt1 ark:x.ark:10")
        second = write(tmp_path, "b", "utt1 ark:y.ark:20", "utt2 ark:y.ark:40")
        merged = merge_unique([first, second], key=FIRST_FIELD)
        assert merged == "utt1 ark:x.ark:10\nutt2 ark:y.ark:40\n"

    def test_the_whole_line_key_keeps_differing_lines(self, tmp_path):
        """
        The difference between the two keys, stated. `ref.wrd.trn` dedupes
        on the whole line, so two lines sharing a first field but differing
        later are both kept.
        """
        first = write(tmp_path, "a", "YES (s-u1)")
        second = write(tmp_path, "b", "YES (s-u2)")
        assert (
            merge_unique([first, second], key=WHOLE_LINE) == "YES (s-u1)\nYES (s-u2)\n"
        )

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "a"
        path.write_text("one\n\n\ntwo\n", encoding="utf-8")
        assert merge_unique([path]) == "one\ntwo\n"

    def test_no_files_gives_empty_text(self):
        assert merge_unique([]) == ""

    def test_empty_files_give_empty_text(self, tmp_path):
        path = tmp_path / "a"
        path.write_text("", encoding="utf-8")
        assert merge_unique([path]) == ""

    def test_the_result_ends_with_one_newline(self, tmp_path):
        merged = merge_unique([write(tmp_path, "a", "one")])
        assert merged.endswith("\n")
        assert not merged.endswith("\n\n")

    def test_an_empty_trn_line_is_not_dropped_by_accident(self, tmp_path):
        """
        An empty hypothesis is written as ` (spk-utt)` -- a line with a
        leading space, not an empty line. It must survive the blank-line
        skip, or the utterance vanishes and `score` refuses the whole
        comparison.
        """
        path = tmp_path / "a"
        path.write_text(" (spk-utt1)\nYES (spk-utt2)\n", encoding="utf-8")
        assert merge_unique([path]) == " (spk-utt1)\nYES (spk-utt2)\n"

    def test_the_counts_the_caller_asserts_still_line_up(self, tmp_path):
        """
        `predict.py` asserts that `ref.wrd.trn` and `output.scp` have the
        same number of lines. Both merges have to agree on which
        utterances survive, or that assertion fires for a reason that has
        nothing to do with the model.
        """
        refs = [
            write(tmp_path, "r.1.trn", "YES (s-u1)", "NO (s-u2)"),
            write(tmp_path, "r.2.trn", "NO (s-u2)", "YES (s-u3)"),
        ]
        scps = [
            write(tmp_path, "o.1.scp", "u1 a:1", "u2 a:2"),
            write(tmp_path, "o.2.scp", "u2 b:2", "u3 b:3"),
        ]
        merged_refs = merge_unique(rank_sorted(refs))
        merged_scps = merge_unique(rank_sorted(scps), key=FIRST_FIELD)
        assert len(merged_refs.splitlines()) == len(merged_scps.splitlines())
