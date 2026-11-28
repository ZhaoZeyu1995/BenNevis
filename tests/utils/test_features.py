"""
Filterbank features in Python, and the gate an 8-bit reference can give.

`read_fbank_conf` needs only the standard library, so its tests are tier
1; computing features needs torch and torchaudio, so those are tier 1b.

**The interesting test here is `TestAgainstTheCapture`, and the reason is
the reference's compression.** `make_fbank.sh` writes compressed
archives, and `CompressedMatrix` quantises each column to 8 bits -- so the
captured features are Kaldi's output *on a grid*, and the closest any
implementation can get to them is half a grid step, a mean of about 0.012
in the log mel domain. Asserting closeness at that tolerance would pass
for an implementation that was genuinely 0.01 wrong.

So the assertion is not closeness. It is that **the captured value is the
nearest grid point to ours**, which holds for 99.97% of entries and fails
immediately for an implementation that is actually wrong -- a gate at
roughly 7e-6 built out of a reference quantised to 0.05. That is what
settled the "1.2% residual" CLAUDE.md chased through six hypotheses, and
`tools/fbank/probe.py` is the independent implementation it was settled
with.

It skips until a capture records the **waveforms**, which
`capture.py` does from the change that found this. Feature extraction was
the one stage whose input was never captured, so until a fresh capture
exists there is nothing to compute features *from*.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json
from pathlib import Path

import pytest

from BenNevis.utils.features import read_fbank_conf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = REPO_ROOT / "tests" / "golden" / "reference"

#: What each recipe's own `conf/fbank.conf` should yield, so a change to
#: one of those files is a visible decision rather than a silent one.
RECIPE_SETTINGS = {
    "egs/yesno/asr": {"num_mel_bins": 80, "sample_frequency": 8000.0},
    "egs/mini_librispeech/asr": {"num_mel_bins": 80, "sample_frequency": 16000.0},
}


class TestReadFbankConf:
    def test_it_starts_from_kaldis_defaults(self, tmp_path):
        conf = tmp_path / "fbank.conf"
        conf.write_text("--num-mel-bins=40\n", encoding="utf-8")
        options = read_fbank_conf(conf)
        assert options["num_mel_bins"] == 40
        # Untouched settings keep Kaldi's values, not torchaudio's --
        # which differ on energy_floor (1.0) and would floor every mel
        # energy at 1 rather than at float epsilon.
        assert options["energy_floor"] == 0.0
        assert options["use_power"] is True
        assert options["window_type"] == "povey"

    def test_comments_and_blank_lines_are_ignored(self, tmp_path):
        conf = tmp_path / "fbank.conf"
        conf.write_text(
            "# a paragraph about dither\n\n--dither=0  # and a trailing note\n",
            encoding="utf-8",
        )
        assert read_fbank_conf(conf)["dither"] == 0.0

    def test_booleans_take_kaldis_spellings(self, tmp_path):
        conf = tmp_path / "fbank.conf"
        conf.write_text("--snip-edges=false\n--use-energy=true\n", encoding="utf-8")
        options = read_fbank_conf(conf)
        assert options["snip_edges"] is False
        assert options["use_energy"] is True

    def test_a_bad_boolean_is_refused(self, tmp_path):
        conf = tmp_path / "fbank.conf"
        conf.write_text("--snip-edges=yes\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not true or false"):
            read_fbank_conf(conf)

    def test_an_unknown_option_is_refused_rather_than_ignored(self, tmp_path):
        """
        Ignoring it would produce features the recipe did not ask for, and
        nothing downstream would notice -- the "wrong rather than broken"
        shape this project keeps meeting.
        """
        conf = tmp_path / "fbank.conf"
        conf.write_text("--num-ceps=13\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not an option"):
            read_fbank_conf(conf)

    def test_dither_above_zero_is_refused(self, tmp_path):
        """
        Kaldi's dither is noise reproducible by its own generator and by
        nothing else, so features computed with it cannot be compared with
        Kaldi's at any useful tolerance. A precondition, not a preference.
        """
        conf = tmp_path / "fbank.conf"
        conf.write_text("--dither=1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="dither"):
            read_fbank_conf(conf)

    def test_an_unparseable_line_names_its_number(self, tmp_path):
        conf = tmp_path / "fbank.conf"
        conf.write_text("--num-mel-bins=80\nnonsense\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            read_fbank_conf(conf)

    @pytest.mark.parametrize("recipe", sorted(RECIPE_SETTINGS))
    def test_each_recipes_configuration_reads_as_expected(self, recipe):
        conf = REPO_ROOT / recipe / "conf" / "fbank.conf"
        if not conf.is_file():
            pytest.skip(f"{recipe} has no conf/fbank.conf")
        options = read_fbank_conf(conf)
        for name, value in RECIPE_SETTINGS[recipe].items():
            assert options[name] == value, f"{recipe}: {name}"
        # Every recipe must have dither off, which `read_fbank_conf`
        # enforces by raising -- so reaching here is the assertion.
        assert options["dither"] == 0.0


torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("torchaudio")

from BenNevis.utils.features import fbank  # noqa: E402


class TestFbank:
    @staticmethod
    def tone(frames=400, rate=8000, freq=440.0):
        t = np.arange(frames) / rate
        return (8000 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_it_returns_frames_by_mel_bins(self):
        out = fbank(
            self.tone(8000), sample_rate=8000, num_mel_bins=80, sample_frequency=8000.0
        )
        assert out.shape[1] == 80
        # snip_edges: 25ms window, 10ms shift at 8 kHz is 200 and 80.
        assert out.shape[0] == 1 + (8000 - 200) // 80

    def test_a_rate_disagreement_is_refused(self):
        """
        Every mel bin edge is derived from the rate, so a mismatch
        silently produces different features -- the kind of error that
        surfaces as a WER nobody can explain.
        """
        with pytest.raises(ValueError, match="silently produce"):
            fbank(self.tone(), sample_rate=16000, sample_frequency=8000.0)

    def test_a_matching_rate_is_accepted(self):
        assert fbank(self.tone(), sample_rate=8000, sample_frequency=8000.0).shape[0]

    def test_one_channel_is_required(self):
        with pytest.raises(ValueError, match="one channel"):
            fbank(np.zeros((2, 4000), dtype=np.float32), sample_frequency=8000.0)

    def test_a_single_channel_axis_is_accepted(self):
        """`kaldiio` can hand back `(1, n)`, so it is unwrapped."""
        flat = fbank(self.tone(), sample_frequency=8000.0)
        shaped = fbank(self.tone()[None, :], sample_frequency=8000.0)
        assert torch.allclose(flat, shaped)

    def test_a_read_only_array_is_handled(self, recwarn):
        """
        `kaldiio` hands back a read-only view of the archive, and torch
        warns once per process about non-writable tensors -- noise in the
        middle of a feature extraction log, and the same reason
        `run/decode_k2.py` copies its log probabilities.

        Asserted on the warning *and* on the answer, because copying to
        silence a warning would be worth nothing if it changed the result.
        """
        samples = self.tone()
        samples.setflags(write=False)
        answer = fbank(samples, sample_frequency=8000.0)
        assert not [
            w for w in recwarn.list if "not writable" in str(w.message)
        ], "torch warned about a non-writable tensor"
        assert torch.allclose(answer, fbank(samples.copy(), sample_frequency=8000.0))

    def test_the_scale_is_kaldis(self):
        """
        Kaldi reads a wav and does **not** normalise, so features are of
        int16-valued samples. A waveform in [-1, 1] gives log energies
        about `log(32768**2)` -- 20.8 -- too small, which is a plausible
        mistake with an implausible symptom.
        """
        loud = fbank(self.tone(), sample_frequency=8000.0)
        quiet = fbank(self.tone() / 32768.0, sample_frequency=8000.0)
        shift = (loud - quiet).mean().item()
        assert 18.0 < shift < 22.0, shift


class TestAgainstTheCapture:
    """
    Gated on the grid, not on closeness. See the module docstring.
    """

    @staticmethod
    def corpora():
        if not REFERENCE.is_dir():
            return []
        found = []
        for directory in sorted(REFERENCE.iterdir()):
            manifest = directory / "manifest.json"
            if not manifest.is_file():
                continue
            arrays = json.loads(manifest.read_text(encoding="utf-8")).get("arrays", {})
            if any(name.startswith("audio/") for name in arrays):
                found.append(directory.name)
        return found

    @pytest.mark.parametrize(
        "corpus",
        corpora.__func__()
        or [
            pytest.param(
                "none",
                marks=pytest.mark.skip(
                    reason="no capture records the waveforms yet; capture.py "
                    "does so from the change that found the compression, so "
                    "this activates on the next capture"
                ),
            )
        ],
    )
    def test_the_captured_value_is_the_nearest_grid_point(self, corpus):
        directory = REFERENCE / corpus
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        recipe = REPO_ROOT / manifest["recipe"]
        options = read_fbank_conf(recipe / "conf" / "fbank.conf")

        compared = 0
        for name, entry in sorted(manifest["arrays"].items()):
            if not name.startswith("audio/"):
                continue
            _kind, dataset, uttid = name.split("/", 2)
            raw = manifest["arrays"].get(f"raw/{dataset}/{uttid}")
            if raw is None:
                continue
            audio = np.load(directory / entry["path"], allow_pickle=True)[uttid]
            reference = np.load(directory / raw["path"], allow_pickle=True)[uttid]
            ours = fbank(audio, sample_rate=entry.get("sample_rate"), **options).numpy()
            width = min(ours.shape[1], reference.shape[1])
            n = min(len(ours), len(reference))
            ours, reference = ours[:n, :width], reference[:n, :width]
            assert n > 0

            wrong = 0
            for column in range(width):
                grid = np.unique(reference[:, column])
                nearest = grid[
                    np.abs(ours[:, column][:, None] - grid[None, :]).argmin(axis=1)
                ]
                wrong += int((nearest != reference[:, column]).sum())
            total = reference.size
            compared += total
            assert wrong / total < 0.01, (
                f"{name}: {wrong} of {total} entries ({100 * wrong / total:.2f}%) "
                f"are not the nearest grid point to ours, so this is not "
                f"Kaldi's filterbank quantised -- it is a different "
                f"computation"
            )
        # A test that compared nothing would pass, which is how the
        # `G.fst` gate in #41 managed to be a test that could not fail.
        assert compared > 0, f"{corpus} recorded audio but nothing was compared"
