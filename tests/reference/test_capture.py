"""
Tests for the capture script, run against a synthetic recipe tree.

The capture only ever runs inside the Kaldi image against a finished recipe,
which makes it exactly the kind of code that is never tested and quietly
wrong when it finally matters. So a fake recipe directory is built here with
real Kaldi-format archives -- `kaldiio` writes them, and it is already a
dependency -- and the script is run against it in process.

That covers everything except `fstprint`, which needs OpenFST. Those
artefacts are expected to be absent here, and the script is required to carry
on and capture the rest rather than fail: a partial capture is more useful
than none, and a missing OpenFST is a property of the machine rather than a
defect.

Marked `torch` for numpy, which tier 1 does not install.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("kaldiio")

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

# Locating an executable is POSIX-shaped: `shutil.which` on Windows wants a
# PATHEXT extension, `os.access(..., X_OK)` is meaningless there because
# Windows has no execute bit, and a `#!` line does not run a file. None of
# that is a defect in the capture, which runs only inside the Linux Kaldi
# image -- the recipes are Linux and macOS only (see CLAUDE.md). The tests
# that do not depend on those semantics still run everywhere.
posix_only = pytest.mark.skipif(
    os.name == "nt",
    reason="locating and running an executable this way is POSIX-only; the "
    "capture runs in the Linux image",
)

import capture  # noqa: E402

from BenNevis.utils.reference import MANIFEST_NAME, Manifest  # noqa: E402

TOPOLOGIES = ["ctc", "2state"]
UTTERANCES = [f"global-{index}" for index in range(8)]


@pytest.fixture
def recipe(tmp_path):
    """
    A recipe directory with everything the capture reads, and nothing else.

    Shaped exactly as egs/yesno/asr is after a finished run: the same file
    names, the same directory layout, the same archive formats.
    """
    import kaldiio
    import numpy as np

    root = tmp_path / "recipe"
    lang = root / "data" / "lang"
    (lang / "phones").mkdir(parents=True)

    (lang / "phones.txt").write_text("<eps> 0\n<SIL> 1\nY 2\nN 3\n")
    (lang / "words.txt").write_text("<eps> 0\nYES 1\nNO 2\n#0 3\n")
    (lang / "oov.txt").write_text("<UNK>\n")
    (lang / "oov.int").write_text("1\n")
    (lang / "phones" / "align_lexicon.int").write_text("1 1 2\n2 2 3\n")
    (lang / "phones" / "disambig.int").write_text("4\n")
    # Deliberately absent: silence.int and optional_silence.int, to check
    # that a missing optional artefact is skipped rather than fatal.

    rng = np.random.default_rng(1368)
    for dataset in ("train_yesno", "test_yesno"):
        data = root / "data" / dataset
        data.mkdir(parents=True)
        features = {
            utterance: rng.standard_normal((20 + index, 4)).astype("float32")
            for index, utterance in enumerate(UTTERANCES)
        }
        kaldiio.save_ark(
            str(data / "feats.cmvn.ark"),
            features,
            scp=str(data / "feats.cmvn.scp"),
        )
        # The pre-CMVN side, deliberately different numbers so a test can
        # tell the two apart rather than passing on either.
        kaldiio.save_ark(
            str(data / "feats.ark"),
            {key: value * 3.0 + 7.0 for key, value in features.items()},
            scp=str(data / "feats.scp"),
        )
        kaldiio.save_ark(
            str(data / "cmvn.ark"),
            {"global": rng.standard_normal((2, 5)).astype("float32")},
            scp=str(data / "cmvn.scp"),
        )

    for topology in TOPOLOGIES:
        (root / "data" / f"lang_test_tg_{topology}").mkdir(parents=True)
        experiment = root / "exp" / f"rnnp-{topology}"
        prediction = experiment / "pred_test_yesno"
        prediction.mkdir(parents=True)
        (prediction / "ref.wrd.trn").write_text(
            "".join(f"YES NO ({utterance})\n" for utterance in UTTERANCES)
        )
        decode = experiment / "dec_test_yesno" / "aw_1.0-ma_5000-bm_32"
        decode.mkdir(parents=True)
        (decode / "hyp.wrd.trn").write_text(
            "".join(f"YES NO ({utterance})\n" for utterance in UTTERANCES)
        )

    return root


def run(recipe_dir, destination, *extra):
    """
    Run the capture and return its exit status.

    `--allow-no-fstprint` because the synthetic recipe has no compiled FSTs
    and this machine need not have OpenFST. The requirement itself is tested
    in TestFstprint, which is where it belongs.
    """
    return capture.main(
        [
            str(recipe_dir),
            str(destination),
            "--corpus",
            "yesno",
            "--datasets",
            "train_yesno",
            "test_yesno",
            "--allow-no-fstprint",
            *extra,
        ]
    )


@pytest.fixture
def captured(recipe, tmp_path):
    """A completed capture: the destination and its manifest."""
    destination = tmp_path / "reference"
    assert run(recipe, destination) == 0
    out = destination / "yesno"
    return out, Manifest.read(out / MANIFEST_NAME)


class TestManifestContents:
    def test_the_manifest_is_written_and_readable(self, captured):
        out, manifest = captured
        assert (out / MANIFEST_NAME).is_file()
        assert manifest.corpus == "yesno"
        assert not manifest.is_empty()

    def test_the_provenance_is_recorded(self, recipe, tmp_path, monkeypatch):
        """
        A reference whose provenance is unknown is not a reference: it must
        say which image produced it.
        """
        monkeypatch.setenv("BENNEVIS_IMAGE", "ghcr.io/x/y@sha256:" + "a" * 64)
        destination = tmp_path / "reference"
        assert run(recipe, destination) == 0
        manifest = Manifest.read(destination / "yesno" / MANIFEST_NAME)
        assert manifest.image.endswith("a" * 64)
        assert manifest.captured

    def test_an_unknown_image_is_said_to_be_unknown(
        self, recipe, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("BENNEVIS_IMAGE", raising=False)
        destination = tmp_path / "reference"
        assert run(recipe, destination) == 0
        manifest = Manifest.read(destination / "yesno" / MANIFEST_NAME)
        assert manifest.image == "unknown"

    def test_the_notes_explain_the_two_kinds_of_entry(self, captured):
        _, manifest = captured
        joined = " ".join(manifest.notes).lower()
        assert "bytes" in joined
        assert "tolerance" in joined


class TestExactArtefacts:
    @pytest.mark.parametrize(
        "name",
        [
            "lang/phones.txt",
            "lang/words.txt",
            "lang/oov.txt",
            "lang/oov.int",
            "lang/phones/align_lexicon.int",
            "lang/phones/disambig.int",
        ],
    )
    def test_the_language_text_is_captured(self, captured, name):
        out, manifest = captured
        assert name in manifest.exact
        assert (out / name).is_file()

    def test_an_absent_optional_artefact_is_skipped(self, captured):
        """
        silence.int was deliberately left out of the fixture. A recipe that
        does not produce it must not fail the capture.
        """
        _, manifest = captured
        assert "lang/phones/silence.int" not in manifest.exact

    def test_the_digest_matches_the_stored_file(self, captured):
        from BenNevis.utils.reference import sha256_file

        out, manifest = captured
        for name, digest in manifest.exact.items():
            assert sha256_file(out / name) == digest, name

    def test_the_content_is_captured_verbatim(self, captured, recipe):
        out, _ = captured
        assert (out / "lang/phones.txt").read_bytes() == (
            recipe / "data" / "lang" / "phones.txt"
        ).read_bytes()

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_transcriptions_are_captured_per_topology(self, captured, topology):
        _, manifest = captured
        assert f"transcriptions/{topology}/ref.wrd.trn" in manifest.exact
        assert f"transcriptions/{topology}/hyp.wrd.trn" in manifest.exact


class TestFeatures:
    def test_both_sides_of_cmvn_are_captured(self, captured):
        """
        The whole reason the capture was extended: a replacement for CMVN or
        for feature extraction needs the input as well as the output.
        """
        _, manifest = captured
        for dataset in ("train_yesno", "test_yesno"):
            assert any(
                name.startswith(f"raw/{dataset}/") for name in manifest.arrays
            ), f"no pre-CMVN features for {dataset}"
            assert any(
                name.startswith(f"features/{dataset}/") for name in manifest.arrays
            ), f"no post-CMVN features for {dataset}"

    def test_the_two_sides_cover_the_same_utterances(self, captured):
        """
        Otherwise they cannot be compared element by element, which is the
        only comparison worth having at a tolerance.
        """
        _, manifest = captured
        for dataset in ("train_yesno", "test_yesno"):
            raw = {
                name.split("/")[-1]
                for name in manifest.arrays
                if name.startswith(f"raw/{dataset}/")
            }
            cooked = {
                name.split("/")[-1]
                for name in manifest.arrays
                if name.startswith(f"features/{dataset}/")
            }
            assert raw == cooked, dataset

    def test_the_two_sides_are_not_the_same_data(self, captured):
        """
        Guards the capture itself: reading feats.scp where feats.cmvn.scp
        was meant, or the reverse, would make every later comparison
        trivially pass.
        """
        import numpy as np

        out, manifest = captured
        name = next(n for n in manifest.arrays if n.startswith("raw/test_yesno/"))
        key = manifest.arrays[name]["key"]
        with np.load(out / "raw/test_yesno.npz") as bundle:
            raw = bundle[key]
        with np.load(out / "features/test_yesno.npz") as bundle:
            cooked = bundle[key]
        assert not np.allclose(raw, cooked)

    def test_a_few_matrices_are_stored_whole(self, captured):
        _, manifest = captured
        whole = [
            name for name in manifest.arrays if name.startswith("features/test_yesno/")
        ]
        assert len(whole) == capture.FULL_FEATURE_UTTERANCES

    def test_every_utterance_has_statistics(self, captured):
        _, manifest = captured
        for dataset in ("train_yesno", "test_yesno"):
            recorded = [
                name
                for name in manifest.stats
                if name.startswith(f"features/{dataset}/")
            ]
            assert len(recorded) == len(UTTERANCES)

    def test_the_stored_arrays_are_the_real_features(self, captured, recipe):
        import kaldiio
        import numpy as np

        out, manifest = captured
        source = dict(
            kaldiio.load_scp_sequential(
                str(recipe / "data" / "test_yesno" / "feats.cmvn.scp")
            )
        )
        for name, record in manifest.arrays.items():
            if not name.startswith("features/test_yesno/"):
                continue
            with np.load(out / record["path"]) as bundle:
                stored = bundle[record["key"]]
            assert np.array_equal(stored, source[record["key"]]), name

    def test_the_recorded_shape_matches_the_array(self, captured):
        import numpy as np

        out, manifest = captured
        for name, record in manifest.arrays.items():
            with np.load(out / record["path"]) as bundle:
                stored = bundle[record["key"]]
            assert list(stored.shape) == record["shape"], name

    def test_the_statistics_describe_the_real_features(self, captured, recipe):
        import kaldiio

        from BenNevis.utils.reference import array_stats

        _, manifest = captured
        source = dict(
            kaldiio.load_scp_sequential(
                str(recipe / "data" / "test_yesno" / "feats.cmvn.scp")
            )
        )
        for utterance, array in source.items():
            recorded = manifest.stats[f"features/test_yesno/{utterance}"]
            assert recorded == pytest.approx(array_stats(array))

    def test_features_carry_the_feature_tolerance(self, captured):
        """
        Not exact: bit-exactness across implementations is not achievable
        for features, and a reference that demanded it would be unusable.
        """
        from BenNevis.utils.reference import FEATURE_TOLERANCE

        _, manifest = captured
        for name in manifest.arrays:
            tolerance = manifest.tolerance_for(name)
            assert tolerance == FEATURE_TOLERANCE, name
            assert not tolerance.is_exact

    def test_cmvn_statistics_are_stored_whole(self, captured):
        _, manifest = captured
        for dataset in ("train_yesno", "test_yesno"):
            assert f"cmvn/{dataset}/global" in manifest.arrays


class TestRobustness:
    def test_a_missing_feature_scp_is_skipped(self, recipe, tmp_path):
        """
        A recipe that stopped before CMVN should still yield whatever it did
        produce.
        """
        (recipe / "data" / "train_yesno" / "feats.cmvn.scp").unlink()
        destination = tmp_path / "reference"
        assert run(recipe, destination) == 0
        manifest = Manifest.read(destination / "yesno" / MANIFEST_NAME)
        assert not any(
            name.startswith("features/train_yesno/") for name in manifest.stats
        )
        assert any(name.startswith("features/test_yesno/") for name in manifest.stats)

    def test_a_directory_with_no_data_is_an_error(self, tmp_path):
        assert run(tmp_path / "nowhere", tmp_path / "reference") == 1

    def test_an_empty_capture_is_an_error(self, tmp_path):
        """
        Reporting success having captured nothing would commit an empty
        reference and quietly make every later comparison vacuous.
        """
        root = tmp_path / "bare"
        (root / "data").mkdir(parents=True)
        assert run(root, tmp_path / "reference") == 1

    def test_the_topologies_are_discovered_from_the_recipe(self, captured):
        _, manifest = captured
        for topology in TOPOLOGIES:
            assert f"transcriptions/{topology}/hyp.wrd.trn" in manifest.exact

    def test_the_topologies_can_be_restricted(self, recipe, tmp_path):
        destination = tmp_path / "reference"
        assert run(recipe, destination, "--topos", "ctc") == 0
        manifest = Manifest.read(destination / "yesno" / MANIFEST_NAME)
        assert "transcriptions/ctc/hyp.wrd.trn" in manifest.exact
        assert "transcriptions/2state/hyp.wrd.trn" not in manifest.exact

    def test_running_twice_gives_the_same_manifest(self, recipe, tmp_path):
        """
        Capture must be deterministic, or a re-capture would appear in review
        as a diff with no meaning.
        """
        first = tmp_path / "a"
        second = tmp_path / "b"
        assert run(recipe, first) == 0
        assert run(recipe, second) == 0

        def without_date(path):
            payload = json.loads((path / "yesno" / MANIFEST_NAME).read_text())
            payload.pop("captured")
            return payload

        assert without_date(first) == without_date(second)


class TestFstprint:
    """
    Finding fstprint, and refusing to produce a capture without it.

    The first two real captures came out with no FSTs at all: fstprint is in
    the Kaldi image but only on the PATH a recipe's path.sh builds, and the
    capture is invoked directly. `shutil.which` found nothing and the code
    carried on quietly, which is how an incomplete reference nearly got
    committed.
    """

    @posix_only
    def test_it_is_found_on_the_path(self, tmp_path, monkeypatch):
        fake = tmp_path / "bin"
        fake.mkdir()
        (fake / "fstprint").write_text("#!/bin/sh\necho\n")
        (fake / "fstprint").chmod(0o755)
        monkeypatch.setenv("PATH", str(fake), prepend=False)
        assert capture.find_fstprint() == str(fake / "fstprint")

    @posix_only
    def test_it_is_found_under_kaldi_root(self, tmp_path, monkeypatch):
        """The case that matters: no PATH, but a Kaldi tree to look in."""
        root = tmp_path / "kaldi"
        binary = root / "tools" / "openfst" / "bin" / "fstprint"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\necho\n")
        binary.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("KALDI_ROOT", str(root))
        assert capture.find_fstprint() == str(binary)

    @posix_only
    def test_a_non_executable_candidate_is_not_used(self, tmp_path, monkeypatch):
        root = tmp_path / "kaldi"
        binary = root / "tools" / "openfst" / "bin" / "fstprint"
        binary.parent.mkdir(parents=True)
        binary.write_text("not executable")
        binary.chmod(0o644)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("KALDI_ROOT", str(root))
        assert capture.find_fstprint() is None

    def test_missing_fstprint_is_an_error_by_default(
        self, recipe, tmp_path, monkeypatch
    ):
        """
        Not a warning. A capture without the lexicon and graph FSTs cannot
        gate the work it exists for, so it must not look like a success.
        """
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("KALDI_ROOT", str(tmp_path / "no-kaldi"))
        status = capture.main([str(recipe), str(tmp_path / "out"), "--corpus", "yesno"])
        assert status == 1

    def test_missing_fstprint_can_be_allowed_explicitly(
        self, recipe, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("KALDI_ROOT", str(tmp_path / "no-kaldi"))
        assert run(recipe, tmp_path / "out") == 0

    @posix_only
    def test_the_fsts_are_captured_when_fstprint_works(
        self, recipe, tmp_path, monkeypatch
    ):
        """
        End to end with a stand-in fstprint, so the wiring from "found the
        binary" to "the manifest has the FST" is exercised rather than
        assumed.
        """
        import sys as _sys

        lang = recipe / "data" / "lang"
        (lang / "L.fst").write_bytes(b"binary")
        (lang / "L_disambig.fst").write_bytes(b"binary")
        for topology in TOPOLOGIES:
            graph = recipe / "data" / f"lang_test_tg_{topology}"
            for name in ("G.fst", "TLG.fst", "LG.fst"):
                (graph / name).write_bytes(b"binary")

        fake = tmp_path / "bin"
        fake.mkdir()
        printer = fake / "fstprint"
        # Prints a one-arc FST, whatever it is given, so the captured text is
        # recognisable and the same for every input.
        printer.write_text(
            f"#!{_sys.executable}\n" "print('0 1 2 3 0.0')\n" "print('1')\n"
        )
        printer.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake))

        destination = tmp_path / "out"
        status = capture.main(
            [
                str(recipe),
                str(destination),
                "--corpus",
                "yesno",
                "--datasets",
                "train_yesno",
                "test_yesno",
            ]
        )
        assert status == 0

        out = destination / "yesno"
        manifest = Manifest.read(out / MANIFEST_NAME)
        assert "lang/L.fst.txt" in manifest.exact
        assert "lang/L_disambig.fst.txt" in manifest.exact
        for topology in TOPOLOGIES:
            assert f"graph/{topology}/TLG.fst.txt" in manifest.exact
        assert (out / "lang" / "L.fst.txt").read_text() == "0 1 2 3 0.0\n1\n"

    @posix_only
    def test_an_expected_fst_that_never_got_built_is_an_error(
        self, recipe, tmp_path, monkeypatch
    ):
        """
        fstprint present, but the recipe produced no L.fst. Reporting
        success here would commit a reference with nothing to compare the
        lexicon against.
        """
        import sys as _sys

        fake = tmp_path / "bin"
        fake.mkdir()
        printer = fake / "fstprint"
        printer.write_text(f"#!{_sys.executable}\nprint('0')\n")
        printer.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake))

        status = capture.main([str(recipe), str(tmp_path / "out"), "--corpus", "yesno"])
        assert status == 1


class TestKaldiToolLocation:
    """`find_kaldi_tool` covers more than one tool now."""

    @posix_only
    @pytest.mark.parametrize("tool", sorted(capture.TOOL_LOCATIONS))
    def test_each_tool_is_found_under_kaldi_root(self, tool, tmp_path, monkeypatch):
        root = tmp_path / "kaldi"
        binary = root / capture.TOOL_LOCATIONS[tool] / tool
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv("KALDI_ROOT", str(root))
        assert capture.find_kaldi_tool(tool) == str(binary)

    def test_fstprint_still_has_its_own_entry_point(self):
        """Kept so the existing call sites and tests read the same."""
        assert "fstprint" in capture.TOOL_LOCATIONS

    def test_copy_feats_is_looked_for_in_featbin(self):
        """
        Where Kaldi puts it. The earlier smoke failure came from guessing a
        directory, so this is stated rather than assumed.
        """
        assert capture.TOOL_LOCATIONS["copy-feats"] == "src/featbin"


@posix_only
class TestReadArchiveThroughCopyFeats:
    """
    Reading an archive via Kaldi's own copy-feats.

    This is what makes a bit-exact comparison possible: `kaldiio`
    decompresses Kaldi's CompressedMatrix slightly differently from Kaldi,
    which capped the CMVN comparison at about 4e-6. Letting Kaldi write an
    uncompressed copy removes that round trip.
    """

    def test_it_reads_through_the_tool_when_one_is_given(self, tmp_path):
        import kaldiio
        import numpy as np

        rng = np.random.default_rng(5)
        wanted = {
            "u1": rng.standard_normal((4, 3)).astype("float32"),
            "u2": rng.standard_normal((6, 3)).astype("float32"),
        }
        kaldiio.save_ark(str(tmp_path / "in.ark"), wanted, scp=str(tmp_path / "in.scp"))

        # A stand-in copy-feats: reads scp:, writes ark:, unchanged. Enough
        # to exercise the plumbing without a Kaldi build.
        tool = tmp_path / "copy-feats"
        tool.write_text(
            f"#!{sys.executable}\n"
            "import sys, kaldiio\n"
            "args = [a for a in sys.argv[1:] if not a.startswith('--')]\n"
            "src = args[0].split(':', 1)[1]\n"
            "dst = args[1].split(':', 1)[1]\n"
            "kaldiio.save_ark(dst, dict(kaldiio.load_scp_sequential(src)))\n"
        )
        tool.chmod(0o755)

        got = capture.read_archive(tmp_path / "in.scp", copy_feats=str(tool))
        assert set(got) == set(wanted)
        for key in wanted:
            assert np.array_equal(got[key], wanted[key])

    def test_it_reads_directly_when_no_tool_is_given(self, tmp_path):
        import kaldiio
        import numpy as np

        wanted = {"u1": np.arange(12, dtype="float32").reshape(4, 3)}
        kaldiio.save_ark(str(tmp_path / "in.ark"), wanted, scp=str(tmp_path / "in.scp"))
        got = capture.read_archive(tmp_path / "in.scp")
        assert np.array_equal(got["u1"], wanted["u1"])

    def test_a_failing_tool_is_not_swallowed(self, tmp_path):
        """
        A capture that silently fell back to the lossy read would look
        complete and compare worse, which is the failure this whole change
        exists to remove.
        """
        import kaldiio
        import numpy as np

        kaldiio.save_ark(
            str(tmp_path / "in.ark"),
            {"u1": np.zeros((2, 2), dtype="float32")},
            scp=str(tmp_path / "in.scp"),
        )
        tool = tmp_path / "copy-feats"
        tool.write_text("#!/bin/sh\nexit 3\n")
        tool.chmod(0o755)

        with pytest.raises(subprocess.CalledProcessError):
            capture.read_archive(tmp_path / "in.scp", copy_feats=str(tool))
