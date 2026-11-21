"""
Tests for BenNevis.utils.data, the Kaldi-format readers.

These functions parse every data directory the toolkit reads, so their edge
cases -- multi-word values, skipped columns, type mappings -- matter more than
their happy path.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

from BenNevis.utils.data import read_dict, read_keys


def write(tmp_path, name, content):
    """Write a fixture file and return its path."""
    path = tmp_path / name
    path.write_text(content)
    return path


class TestReadDict:
    """read_dict parses '<key> <value>...' files."""

    def test_single_value(self, tmp_path):
        path = write(tmp_path, "utt2spk", "utt1 spk1\nutt2 spk2\n")
        assert read_dict(path) == {"utt1": "spk1", "utt2": "spk2"}

    def test_joins_multiple_values_with_single_spaces(self, tmp_path):
        """A transcript is many tokens; they must come back as one string."""
        path = write(tmp_path, "text", "utt1 HELLO   THERE  WORLD\n")
        assert read_dict(path) == {"utt1": "HELLO THERE WORLD"}

    def test_value_start_skips_columns(self, tmp_path):
        """align_lexicon.int repeats the key, so values start at column 2."""
        path = write(tmp_path, "align_lexicon.int", "5 5 11 12 13\n")
        assert read_dict(path, value_start=2) == {"5": "11 12 13"}

    def test_mapping_transforms_the_value(self, tmp_path):
        path = write(tmp_path, "utt2dur", "utt1 1.25\nutt2 3.5\n")
        assert read_dict(path, mapping=float) == {"utt1": 1.25, "utt2": 3.5}

    def test_key_mapping_transforms_the_key(self, tmp_path):
        path = write(tmp_path, "align_lexicon.int", "5 5 11 12\n")
        out = read_dict(
            path,
            value_start=2,
            key_mapping=int,
            mapping=lambda x: [int(i) for i in x.split()],
        )
        assert out == {5: [11, 12]}

    def test_blank_line_raises_indexerror(self, tmp_path):
        """
        Characterisation test, not an endorsement: a blank line currently
        raises IndexError rather than being skipped. Kaldi files do sometimes
        carry a trailing blank line, so this is a robustness gap -- but it is
        existing behaviour and is pinned here rather than changed.
        """
        path = write(tmp_path, "utt2spk", "utt1 spk1\n\n")
        with pytest.raises(IndexError):
            read_dict(path)

    def test_rejects_a_line_with_no_value(self, tmp_path):
        """A key with no value is a corrupt data directory, not an empty string."""
        path = write(tmp_path, "utt2spk", "utt1\n")
        with pytest.raises(AssertionError):
            read_dict(path)

    def test_later_duplicate_key_wins(self, tmp_path):
        """Documents existing behaviour: duplicates silently overwrite."""
        path = write(tmp_path, "utt2spk", "utt1 spk1\nutt1 spk2\n")
        assert read_dict(path) == {"utt1": "spk2"}


class TestReadKeys:
    """read_keys returns the first column, preserving file order."""

    def test_returns_keys_in_order(self, tmp_path):
        path = write(tmp_path, "wav.scp", "b x\na y\nc z\n")
        assert read_keys(path) == ["b", "a", "c"]

    def test_works_on_a_symbol_table(self, tmp_path):
        path = write(tmp_path, "phones.txt", "<eps> 0\nAA 1\n#0 2\n")
        assert read_keys(path) == ["<eps>", "AA", "#0"]

    def test_keeps_duplicates(self, tmp_path):
        """Unlike read_dict, read_keys does not deduplicate."""
        path = write(tmp_path, "segments", "utt1 a\nutt1 b\n")
        assert read_keys(path) == ["utt1", "utt1"]

    def test_single_column_file(self, tmp_path):
        """disambig.int has one column per line."""
        path = write(tmp_path, "disambig.int", "5\n6\n")
        assert read_keys(path) == ["5", "6"]

    def test_blank_line_raises_indexerror(self, tmp_path):
        """Same robustness gap as read_dict; pinned, not endorsed."""
        path = write(tmp_path, "wav.scp", "utt1 a.wav\n\n")
        with pytest.raises(IndexError):
            read_keys(path)
