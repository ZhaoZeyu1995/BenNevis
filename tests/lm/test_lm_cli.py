"""
`python3 -m BenNevis.lm lexicon-fst`, which writes `k2/L.fst.txt`.

Tier 1: `BenNevis.lm.lexicon` needs nothing but the standard library, so
the thing that takes OpenFST off the training path is itself testable in
the cheapest tier.

The arithmetic is tested in `test_lexicon.py`; this is about the command
line `prepare_topo.sh` calls -- that it writes to stdout, that the two
settings reach the builder, and that the optional-silence phone is read
from beside the lexicon rather than assumed to be `SIL`. The last one
matters because a recipe whose silence phone is named something else would
otherwise get a lexicon that composes and decodes and is quietly wrong.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from BenNevis.lm.__main__ import main, optional_silence, read_table

#: Two words over three phones, with a silence phone that is deliberately
#: not called SIL.
LEXICON = "HELLO h e l\nWORLD w o\n!SIL sp\n"
WORDS = "<eps> 0\nHELLO 1\nWORLD 2\n!SIL 3\n#0 4\n"
PHONES = "<eps> 0\nh 1\ne 2\nl 3\nw 4\no 5\nsp 6\n#0 7\n"


@pytest.fixture
def dictionary(tmp_path):
    (tmp_path / "lexicon.txt").write_text(LEXICON, encoding="utf-8")
    (tmp_path / "words.txt").write_text(WORDS, encoding="utf-8")
    (tmp_path / "phones.txt").write_text(PHONES, encoding="utf-8")
    return tmp_path


def run(*argv):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(list(argv))
    assert code == 0
    return buffer.getvalue()


class TestItWritesAnFst:
    def test_it_prints_openfst_text_to_stdout(self, dictionary):
        out = run(
            "lexicon-fst",
            str(dictionary / "lexicon.txt"),
            str(dictionary / "words.txt"),
            str(dictionary / "phones.txt"),
        )
        lines = [line for line in out.splitlines() if line.strip()]
        assert lines, "nothing was written"
        for line in lines:
            fields = line.split()
            # An arc is four or five fields; a final state is one or two.
            assert len(fields) in (1, 2, 4, 5), line
            assert all(
                field.lstrip("-").replace(".", "", 1).isdigit() for field in fields
            ), (
                f"{line!r} is not integer-labelled; the symbol tables were "
                f"not applied"
            )

    def test_the_silence_probability_changes_the_weights(self, dictionary):
        without = run(
            "lexicon-fst",
            str(dictionary / "lexicon.txt"),
            str(dictionary / "words.txt"),
            str(dictionary / "phones.txt"),
        )
        with_silence = run(
            "lexicon-fst",
            str(dictionary / "lexicon.txt"),
            str(dictionary / "words.txt"),
            str(dictionary / "phones.txt"),
            "--sil-prob",
            "0.5",
            # This dictionary's silence phone is `sp`, and there is no
            # optional_silence.txt beside it, so the default of SIL would
            # not be in phones.txt -- which is the error the next class
            # asserts and is not what is being tested here.
            "--sil-phone",
            "sp",
        )
        assert without != with_silence
        # -log(0.5) in float32, as fstprint writes it. With sil_prob 0.0
        # every weight is zero and fstprint omits it entirely, which is why
        # the two files differ in shape and not only in numbers.
        assert "0.693147182" in with_silence

    def test_position_dependence_is_a_flag_that_does_something(self, tmp_path):
        """
        With suffixed phones the plain run cannot build at all, which is
        the asymmetry that made a builder ignoring this reproduce yesno's
        bytes and fail on mini-librispeech.
        """
        (tmp_path / "lexicon.txt").write_text("HI h i\n", encoding="utf-8")
        (tmp_path / "words.txt").write_text("<eps> 0\nHI 1\n", encoding="utf-8")
        (tmp_path / "phones.txt").write_text(
            "<eps> 0\nh_B 1\ni_E 2\nSIL 3\n", encoding="utf-8"
        )
        out = run(
            "lexicon-fst",
            str(tmp_path / "lexicon.txt"),
            str(tmp_path / "words.txt"),
            str(tmp_path / "phones.txt"),
            "--position-dependent-phones",
        )
        assert out.strip()
        with pytest.raises(KeyError):
            run(
                "lexicon-fst",
                str(tmp_path / "lexicon.txt"),
                str(tmp_path / "words.txt"),
                str(tmp_path / "phones.txt"),
            )


class TestTheSilencePhone:
    def test_it_is_read_from_beside_the_lexicon(self, dictionary):
        (dictionary / "optional_silence.txt").write_text("sp\n", encoding="utf-8")
        assert optional_silence(dictionary / "lexicon.txt", None) == "sp"

    def test_an_explicit_phone_wins(self, dictionary):
        (dictionary / "optional_silence.txt").write_text("sp\n", encoding="utf-8")
        assert optional_silence(dictionary / "lexicon.txt", "SIL") == "SIL"

    def test_it_falls_back_to_sil(self, dictionary):
        assert optional_silence(dictionary / "lexicon.txt", None) == "SIL"

    def test_an_empty_file_falls_back_rather_than_yielding_nothing(self, dictionary):
        (dictionary / "optional_silence.txt").write_text("\n", encoding="utf-8")
        assert optional_silence(dictionary / "lexicon.txt", None) == "SIL"

    def test_it_reaches_the_weights(self, dictionary):
        """
        A silence phone that is not in `phones.txt` must fail, which is how
        this is shown to be used rather than merely computed.
        """
        (dictionary / "optional_silence.txt").write_text("nope\n", encoding="utf-8")
        with pytest.raises(KeyError):
            run(
                "lexicon-fst",
                str(dictionary / "lexicon.txt"),
                str(dictionary / "words.txt"),
                str(dictionary / "phones.txt"),
                "--sil-prob",
                "0.5",
            )


class TestReadTable:
    def test_it_ignores_lines_that_are_not_pairs(self, tmp_path):
        path = tmp_path / "t.txt"
        path.write_text("<eps> 0\n\nbad\na 1 2\nb 2\n", encoding="utf-8")
        assert read_table(path) == {"<eps>": 0, "b": 2}


class TestTheCommandLineItself:
    def test_an_unknown_command_is_refused(self):
        with pytest.raises(SystemExit):
            main(["nonsense"])

    def test_no_command_is_refused(self):
        with pytest.raises(SystemExit):
            main([])

    def test_it_is_runnable_as_a_module(self, dictionary):
        """
        `prepare_topo.sh` invokes it as `python3 -m BenNevis.lm`, so the
        module has to be executable that way and not only importable --
        which is a different thing and is what a missing `__main__` breaks.
        """
        import subprocess

        if sys.platform == "win32":
            pytest.skip("the recipes are POSIX only")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "BenNevis.lm",
                "lexicon-fst",
                str(dictionary / "lexicon.txt"),
                str(dictionary / "words.txt"),
                str(dictionary / "phones.txt"),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
