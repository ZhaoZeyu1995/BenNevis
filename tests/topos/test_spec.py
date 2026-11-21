"""
Tests for loading and validating topology specifications.

The point of moving topologies into data is that a mistake becomes a loading
error rather than a subtly wrong FST. These tests pin that: every way of
writing an inconsistent spec must be rejected at load time, with a message
that says what is wrong.

The most important of these is the token/arc agreement check. The token
inventory determines the neural network's output layer size, so a spec whose
arcs use a token the inventory does not list would train a network that
cannot express its own topology -- and would fail much later, during graph
composition, with nothing pointing back here.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import textwrap
from pathlib import Path

import pytest

from BenNevis.topos import spec

MINIMAL = """
name: toy
summary: a toy topology
tokens_per_phone: ["{phone}_0", "{phone}_1"]
variants:
  lang:
    stride: 2
    arcs:
      - [hub, s0, "{phone}_0", "{phone}"]
      - [s0, s1, "{phone}_1", "<eps>"]
      - [s1, hub, "<eps>", "<eps>"]
"""


def write(tmp_path: Path, text: str, name: str = "toy.yaml") -> Path:
    """Write a spec file and return its path."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


class TestLoading:
    def test_a_minimal_spec_loads(self, tmp_path):
        topology = spec.load_spec(write(tmp_path, MINIMAL))
        assert topology.name == "toy"
        assert topology.tokens_per_phone == ("{phone}_0", "{phone}_1")
        assert topology.variants["lang"].stride == 2

    def test_states_are_resolved_to_offsets(self, tmp_path):
        arcs = spec.load_spec(write(tmp_path, MINIMAL)).variants["lang"].arcs
        assert arcs[0].src is None  # the hub
        assert arcs[0].dst == 0
        assert arcs[2].dst is None

    def test_tokens_for_phone_expands_the_templates(self, tmp_path):
        topology = spec.load_spec(write(tmp_path, MINIMAL))
        assert topology.tokens_for_phone("AA") == ["AA_0", "AA_1"]

    def test_defaults_are_the_common_case(self, tmp_path):
        topology = spec.load_spec(write(tmp_path, MINIMAL))
        assert topology.blank is True
        assert topology.skip_phones == ("<eps>", "<SIL>")
        assert topology.variants["lang"].finals == spec.FINALS_HUB
        assert topology.variants["lang"].disambig == spec.DISAMBIG_SELF_LOOP


class TestRejections:
    """Each of these would otherwise produce a quietly wrong topology."""

    def test_an_arc_label_outside_the_token_inventory_is_rejected(self, tmp_path):
        bad = MINIMAL.replace(
            '[s0, s1, "{phone}_1", "<eps>"]', '[s0, s1, "{phone}_7", "<eps>"]'
        )
        with pytest.raises(ValueError, match="not one of the tokens"):
            spec.load_spec(write(tmp_path, bad))

    def test_a_state_outside_the_phone_block_is_rejected(self, tmp_path):
        bad = MINIMAL.replace("[s0, s1,", "[s0, s9,")
        with pytest.raises(ValueError, match="outside the phone block"):
            spec.load_spec(write(tmp_path, bad))

    def test_an_unparseable_state_reference_is_rejected(self, tmp_path):
        bad = MINIMAL.replace("[hub, s0,", "[start, s0,")
        with pytest.raises(ValueError, match="neither 'hub' nor 'sN'"):
            spec.load_spec(write(tmp_path, bad))

    def test_an_arc_with_the_wrong_number_of_fields_is_rejected(self, tmp_path):
        bad = MINIMAL.replace('[s1, hub, "<eps>", "<eps>"]', '[s1, hub, "<eps>"]')
        with pytest.raises(ValueError, match="exactly 4 fields"):
            spec.load_spec(write(tmp_path, bad))

    def test_an_epsilon_input_in_a_k2_variant_is_rejected(self, tmp_path):
        """
        The k2 variant's input symbols *are* the network's outputs, which
        cannot include <eps>. A spec that breaks this would build a graph the
        network can never traverse.
        """
        bad = MINIMAL + (
            "  k2:\n"
            "    stride: 2\n"
            "    disambig: skip\n"
            "    arcs:\n"
            '      - [hub, s0, "{phone}_0", "{phone}"]\n'
            '      - [s0, hub, "<eps>", "<eps>"]\n'
        )
        with pytest.raises(ValueError, match="epsilon-free on the input side"):
            spec.load_spec(write(tmp_path, bad))

    def test_a_zero_stride_is_rejected(self, tmp_path):
        bad = MINIMAL.replace("stride: 2", "stride: 0")
        with pytest.raises(ValueError, match="stride must be at least 1"):
            spec.load_spec(write(tmp_path, bad))

    def test_an_unknown_disambig_mode_is_rejected(self, tmp_path):
        bad = MINIMAL.replace("stride: 2", "stride: 2\n    disambig: maybe")
        with pytest.raises(ValueError, match="disambig must be"):
            spec.load_spec(write(tmp_path, bad))

    def test_an_unknown_finals_mode_is_rejected(self, tmp_path):
        bad = MINIMAL.replace("stride: 2", "stride: 2\n    finals: everything")
        with pytest.raises(ValueError, match="finals must be"):
            spec.load_spec(write(tmp_path, bad))

    def test_a_spec_with_no_variants_is_rejected(self, tmp_path):
        bad = "name: toy\ntokens_per_phone: ['{phone}']\nvariants: {}\n"
        with pytest.raises(ValueError, match="no variants"):
            spec.load_spec(write(tmp_path, bad))

    def test_a_spec_with_no_tokens_is_rejected(self, tmp_path):
        bad = MINIMAL.replace(
            'tokens_per_phone: ["{phone}_0", "{phone}_1"]', "tokens_per_phone: []"
        )
        with pytest.raises(ValueError, match="derives no tokens"):
            spec.load_spec(write(tmp_path, bad))

    def test_a_variant_with_no_arcs_is_rejected(self, tmp_path):
        bad = """
        name: toy
        tokens_per_phone: ["{phone}"]
        variants:
          lang:
            stride: 1
            arcs: []
        """
        with pytest.raises(ValueError, match="no per-phone arcs"):
            spec.load_spec(write(tmp_path, bad))

    def test_a_missing_required_key_names_itself(self, tmp_path):
        bad = MINIMAL.replace("tokens_per_phone:", "tokens:")
        with pytest.raises(ValueError, match="missing required key 'tokens_per_phone'"):
            spec.load_spec(write(tmp_path, bad))

    def test_a_non_mapping_spec_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mapping at the top level"):
            spec.load_spec(write(tmp_path, "- just\n- a\n- list\n"))


class TestBundledSpecs:
    """The specs shipped with the package."""

    def test_the_expected_topologies_are_available(self):
        assert set(spec.available()) == {
            "ctc",
            "mmictc",
            "mmictc-1",
            "2state",
            "2state-1",
            "3state-skip",
            "3state-skip-1",
            "3state-skip-2",
        }

    @pytest.mark.parametrize("name", spec.available())
    def test_every_bundled_spec_loads_and_validates(self, name):
        topology = spec.load(name)
        topology.validate()
        assert topology.name == name

    @pytest.mark.parametrize("name", spec.available())
    def test_every_bundled_spec_has_both_variants(self, name):
        """
        The pipeline needs both: T.fst for decoding and k2/T.fst for training.
        A topology offering only one would break prepare_topo.sh at whichever
        step came second.
        """
        assert set(spec.load(name).variants) == {"lang", "k2"}

    @pytest.mark.parametrize("name", spec.available())
    def test_every_bundled_spec_has_a_summary(self, name):
        """The summary is the only documentation of what the machine looks like."""
        assert spec.load(name).summary.strip()

    def test_an_unknown_name_lists_what_is_available(self):
        with pytest.raises(FileNotFoundError, match="available:"):
            spec.load("no-such-topology")

    def test_asking_for_a_missing_variant_says_what_exists(self):
        with pytest.raises(KeyError, match="has no 'nope' variant"):
            spec.load("ctc").variant("nope")
