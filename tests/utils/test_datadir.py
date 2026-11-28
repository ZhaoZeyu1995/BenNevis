"""
Keeping a data directory consistent, and the rules that are easy to drop.

Tier 1: `BenNevis.utils.datadir` needs nothing but the standard library,
which is the point of putting the logic there rather than in the entry
point. This is the stage of a recipe least likely to have a test and most
able to change a corpus without anyone noticing -- yesno's test set was
29 utterances while claiming 31 for exactly this reason.

Four of Kaldi's rules are reproduced here and none is guessable from a
casual reading of `fix_data_dir.sh`:

* a speaker absent from `cmvn.scp` loses **all of its utterances**,
  because `filter_speakers` filters the speaker list by `cmvn.scp` rather
  than the other way about and then re-derives `utt2spk`;
* `feats.cmvn.scp` is rewritten but must **not** narrow the surviving
  list, the way Kaldi treats `vad.scp` -- it is written by a later stage,
  so on a re-run a stale copy would otherwise delete the new utterances;
* `utt2num_frames` entries of zero are dropped rather than kept at zero;
* sorting is by **bytes**, since the script exports `LC_ALL=C`.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

from BenNevis.utils.datadir import (
    DataDirError,
    fix,
    read_table,
    sort_unique,
    write_table,
)


def build(tmp_path, **files):
    """
    A data directory from a mapping of file name to its lines.

    Returns the directory, so a test reads as "this directory, fixed,
    looks like that".
    """
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        (data / name).write_text("".join(f"{line}\n" for line in lines), "utf-8")
    return data


def keys(path):
    return [key for key, _rest in read_table(path)]


def three_speakers(tmp_path, **overrides):
    """
    Two utterances each for three speakers, every file consistent.

    The baseline the tests perturb one thing at a time from.
    """
    utterances = [f"{spk}-{n}" for spk in ("spkA", "spkB", "spkC") for n in (1, 2)]
    files = {
        "utt2spk": [f"{utt} {utt.split('-')[0]}" for utt in utterances],
        "spk2utt": [f"{spk} {spk}-1 {spk}-2" for spk in ("spkA", "spkB", "spkC")],
        "wav.scp": [f"{utt} ./corpus/{utt}.wav" for utt in utterances],
        "text": [f"{utt} HELLO" for utt in utterances],
        "feats.scp": [
            f"{utt} /ark/feats.ark:{i * 100}" for i, utt in enumerate(utterances)
        ],
        "utt2num_frames": [f"{utt} 100" for utt in utterances],
        "cmvn.scp": [
            f"{spk} /ark/cmvn.ark:{i * 10}"
            for i, spk in enumerate(("spkA", "spkB", "spkC"))
        ],
    }
    files.update(overrides)
    return build(tmp_path, **files)


class TestTheTableHelpers:
    def test_a_sort_is_by_bytes_not_by_locale(self, tmp_path):
        """
        `fix_data_dir.sh` exports `LC_ALL=C`. Under a locale where case or
        punctuation is folded, `README~` sorts among the digits -- which
        is the shape of the bug that put two README files into yesno's
        test set.
        """
        rows = [("README~", "a"), ("0_1", "b"), ("Z", "c"), ("_x", "d")]
        assert [key for key, _ in sort_unique(rows)] == ["0_1", "README~", "Z", "_x"]

    def test_a_repeated_key_keeps_one_line(self):
        rows = [("b", "2"), ("a", "1"), ("a", "9")]
        assert sort_unique(rows) == [("a", "1"), ("b", "2")]

    def test_a_wav_scp_command_survives_a_round_trip(self, tmp_path):
        """
        A `wav.scp` entry is a shell command -- mini-librispeech's are
        `flac -c -d -s ./corpus/... |` -- so reformatting the remainder
        would change what the recipe runs.
        """
        command = "flac -c -d -s ./corpus/LibriSpeech/a/b/c.flac |"
        path = tmp_path / "wav.scp"
        write_table(path, [("utt", command)])
        assert read_table(path) == [("utt", command)]
        assert path.read_text("utf-8") == f"utt {command}\n"

    def test_an_empty_transcription_is_kept(self, tmp_path):
        """
        A line holding only a key is an empty transcription, which
        `fix_data_dir.sh` tolerates -- `validate_data_dir.sh` is what
        complains. Refusing it would reject a directory Kaldi accepts.
        """
        path = tmp_path / "text"
        path.write_text("utt1 HELLO\nutt2\n", "utf-8")
        assert read_table(path) == [("utt1", "HELLO"), ("utt2", "")]
        write_table(path, read_table(path))
        assert path.read_text("utf-8") == "utt1 HELLO\nutt2\n"


class TestItFiltersOnUtterances:
    def test_an_utterance_without_features_is_dropped_everywhere(self, tmp_path):
        """
        The case the stage exists for: feature extraction failed for one
        utterance, so it has a transcription and no features. Leaving it
        in `text` is how a test set comes to claim more than it has.
        """
        data = three_speakers(tmp_path)
        rows = [row for row in read_table(data / "feats.scp") if row[0] != "spkB-1"]
        write_table(data / "feats.scp", rows)

        summary = fix(data)

        assert summary["kept"] == 5
        assert summary["dropped"] == 1
        for name in ("utt2spk", "text", "wav.scp", "utt2num_frames", "feats.scp"):
            assert "spkB-1" not in keys(data / name), name
        assert keys(data / "spk2utt") == ["spkA", "spkB", "spkC"]
        assert read_table(data / "spk2utt")[1] == ("spkB", "spkB-2")

    def test_a_zero_frame_utterance_is_dropped(self, tmp_path):
        """
        `awk '{ if (NF == 2 && $2 > 0) { print }}'`: an utterance of zero
        frames is not something to train on, and keeping it at zero would
        reach the collate function as an empty matrix.
        """
        data = three_speakers(tmp_path)
        rows = [
            (utt, "0" if utt == "spkC-2" else frames)
            for utt, frames in read_table(data / "utt2num_frames")
        ]
        write_table(data / "utt2num_frames", rows)

        assert fix(data)["kept"] == 5
        assert "spkC-2" not in keys(data / "text")

    def test_a_stale_normalised_archive_does_not_delete_utterances(self, tmp_path):
        """
        `feats.cmvn.scp` is written by the *next* stage, so on a re-run it
        is behind the others. Kaldi treats `vad.scp` exactly this way --
        rewritten, never intersected -- and putting this file in the
        intersection instead would silently shrink a corpus to whatever
        the previous run happened to normalise.
        """
        data = three_speakers(
            tmp_path,
            **{"feats.cmvn.scp": ["spkA-1 /ark/old.ark:0"]},
        )

        summary = fix(data)

        assert summary["kept"] == 6, "a stale feats.cmvn.scp narrowed the corpus"
        assert keys(data / "feats.cmvn.scp") == ["spkA-1"]

    def test_losing_every_utterance_is_fatal(self, tmp_path):
        data = three_speakers(tmp_path, **{"feats.scp": []})
        with pytest.raises(DataDirError, match="no utterances remained"):
            fix(data)


class TestItFiltersOnSpeakers:
    def test_a_speaker_without_statistics_loses_its_utterances(self, tmp_path):
        """
        The reversed-argument rule. `filter_speakers` runs
        `filter_file $f $tmpdir/speakers`, which filters the **speaker
        list** by `cmvn.scp`, and then rebuilds `utt2spk` from the
        surviving `spk2utt`. So the direction of that one call decides
        whether a speaker whose statistics failed keeps its utterances,
        and it does not.
        """
        data = three_speakers(tmp_path)
        rows = [row for row in read_table(data / "cmvn.scp") if row[0] != "spkB"]
        write_table(data / "cmvn.scp", rows)

        summary = fix(data)

        assert summary["speakers"] == 2
        assert summary["kept"] == 4, "spkB's utterances outlived its statistics"
        assert keys(data / "utt2spk") == ["spkA-1", "spkA-2", "spkC-1", "spkC-2"]
        assert keys(data / "cmvn.scp") == ["spkA", "spkC"]

    def test_spk2utt_is_derived_rather_than_filtered(self, tmp_path):
        """
        "throughout this program, we regard utt2spk as primary and spk2utt
        as derived" -- so a wrong `spk2utt` on the way in is corrected
        rather than believed.
        """
        data = three_speakers(
            tmp_path,
            **{"spk2utt": ["spkA spkA-1", "spkB spkB-1 spkB-2 ghost", "spkC spkC-2"]},
        )

        fix(data)

        assert read_table(data / "spk2utt") == [
            ("spkA", "spkA-1 spkA-2"),
            ("spkB", "spkB-1 spkB-2"),
            ("spkC", "spkC-1 spkC-2"),
        ]

    def test_losing_every_speaker_is_fatal(self, tmp_path):
        data = three_speakers(tmp_path, **{"cmvn.scp": []})
        with pytest.raises(DataDirError, match="no speaker survived"):
            fix(data)


class TestItSortsWhatItLeaves:
    def test_an_unsorted_directory_comes_back_sorted(self, tmp_path):
        data = three_speakers(tmp_path)
        rows = list(reversed(read_table(data / "text")))
        write_table(data / "text", rows)

        said = []
        fix(data, report=said.append)

        assert keys(data / "text") == sorted(keys(data / "text"))
        assert any("not sorted" in line for line in said)

    def test_the_originals_are_kept(self, tmp_path):
        """
        Kaldi's `.backup`, which is what makes an over-eager filter
        inspectable rather than a re-preparation.
        """
        data = three_speakers(tmp_path)
        before = (data / "text").read_text("utf-8")
        write_table(data / "feats.scp", read_table(data / "feats.scp")[:2])

        fix(data)

        assert (data / ".backup" / "text").read_text("utf-8") == before
        assert (data / "text").read_text("utf-8") != before


class TestItRefusesWhatItCannotHandle:
    @pytest.mark.parametrize("name", ["segments", "vad.scp", "utt2dur", "spk2gender"])
    def test_a_file_it_does_not_filter_is_refused(self, tmp_path, name):
        """
        Leaving one of these unfiltered while filtering everything around
        it is worse than refusing: with `segments` present, `wav.scp` is
        indexed by *recording*, so filtering it against an utterance list
        would empty it.
        """
        data = three_speakers(tmp_path)
        (data / name).write_text("something\n", "utf-8")
        with pytest.raises(DataDirError, match=name.replace(".", r"\.")):
            fix(data)

    def test_a_missing_utt2spk_is_refused(self, tmp_path):
        data = three_speakers(tmp_path)
        (data / "utt2spk").unlink()
        with pytest.raises(DataDirError, match="utt2spk"):
            fix(data)

    def test_a_missing_directory_is_refused(self, tmp_path):
        with pytest.raises(DataDirError, match="no such directory"):
            fix(tmp_path / "absent")


class TestNothingChangesWhenNothingIsWrong:
    def test_a_consistent_directory_is_left_alone(self, tmp_path):
        """
        The other half of the filtering tests: a stage that quietly
        dropped an utterance from a healthy corpus would be far worse
        than one that dropped none from a broken one.
        """
        data = three_speakers(tmp_path)
        before = {
            name: (data / name).read_text("utf-8")
            for name in ("utt2spk", "text", "wav.scp", "feats.scp", "cmvn.scp")
        }

        summary = fix(data)

        assert summary == {
            "kept": 6,
            "dropped": 0,
            "speakers": 3,
            "files": [
                "utt2spk",
                "feats.scp",
                "text",
                "wav.scp",
                "utt2num_frames",
                "spk2utt",
                "cmvn.scp",
            ],
        }
        for name, text in before.items():
            assert (data / name).read_text("utf-8") == text, name
