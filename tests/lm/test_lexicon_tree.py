"""
The lexicon as a prefix tree, and why it is equivalent to the flower.

Tier 1: building the arcs and searching them needs nothing but the
standard library, which is what makes a functional comparison affordable
here rather than only in a nightly.

**This is the gate that replaces byte-exactness, and that is a deliberate
trade.** `make_lexicon_fst.py` builds a *flower*: a fresh chain of states
per entry hanging off the loop state, with the word on the first arc. Our
builder reproduces its bytes exactly for both corpora, which was worth
having -- but the flower is the measured cause of the decoding gap. On
mini-librispeech the loop state carries **206,510 arcs**, and CLAUDE.md's
profiling found "208,941 arcs over 106 distinct input labels -- 1,950
parallel arcs per label -- and 366 states like it hold 52.5% of the whole
graph".

A prefix tree is the shape determinisation would produce, and it is
cheaper to build directly than to determinise 43 million arcs: measured
on mini-librispeech's 206,510 entries, **1,333,084 arcs become 469,947,
1,126,575 states become 263,438, and no state anywhere has more than 195
arcs**.

An FST's encoding is arbitrary; its function is not. So the tree is
checked the way CLAUDE.md says an FST should be -- "the same weight for
every word sequence" -- against the flower, which remains in the code as
the oracle and still reproduces Kaldi's bytes.

Two properties of the tree are forced rather than chosen, and both are
pinned below. The **word rides the last arc**, because the first is
shared; measured, a word becomes unambiguous only 5.90 phones into a mean
pronunciation of 6.46, so no earlier placement exists. And the **final
arc is not shared**, since it carries the word, which is why the saving
is 2.84x and not the 3.37x a fully shared tree would give.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import heapq
import random
from collections import defaultdict
from pathlib import Path

import pytest

from BenNevis.lm.lexicon import build_lexicon_arcs, position_dependent, read_lexicon

REFERENCE = Path(__file__).resolve().parent.parent / "golden" / "reference"

#: What each corpus's `prepare_lang.sh` call was given, which is also what
#: `tests/recipes/test_lexicon_settings.py` pins against the recipes.
SETTINGS = {
    "yesno": {"position_dependent_phones": False, "sil_prob": 0.0},
    "mini_librispeech": {"position_dependent_phones": True, "sil_prob": 0.5},
}


def corpora():
    if not REFERENCE.is_dir():
        return []
    return sorted(
        path.name
        for path in REFERENCE.iterdir()
        if (path / "dict" / "lexicon.txt").is_file()
        and (path / "lang" / "words.txt").is_file()
        and path.name in SETTINGS
    )


def silence_phone(corpus):
    """
    The optional-silence phone, from the dictionary.

    Read rather than defaulted: `build_lexicon_arcs` defaults to `SIL`
    and **yesno's is `<SIL>`**, so overriding its silence probability
    without this raises a KeyError about a phone that is simply spelled
    differently. `prepare_lang.sh` reads `optional_silence.txt` for the
    same reason.
    """
    path = REFERENCE / corpus / "dict" / "optional_silence.txt"
    return path.read_text(encoding="utf-8").split()[0]


def table(corpus, name):
    path = REFERENCE / corpus / "lang" / name
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            out[fields[0]] = int(fields[1])
    return out


_BUILT = {}
_INDEXED = {}


def built(corpus, structure, sil_prob=None):
    """
    The arcs for one corpus and structure, memoised.

    mini-librispeech's flower is 1.3 million arcs and four of these tests
    want it; building it once keeps tier 1 instant, which is the same
    reason `test_prepare_lang.py` memoises.
    """
    settings = dict(SETTINGS[corpus])
    if sil_prob is not None:
        settings["sil_prob"] = sil_prob
    key = (corpus, structure, settings["sil_prob"])
    if key not in _BUILT:
        _BUILT[key] = build_lexicon_arcs(
            read_lexicon(REFERENCE / corpus / "dict" / "lexicon.txt"),
            table(corpus, "words.txt"),
            table(corpus, "phones.txt"),
            sil_phone=silence_phone(corpus),
            structure=structure,
            **settings,
        )
    return _BUILT[key]


def states_of(arcs):
    """How many distinct states the arcs mention."""
    return len({a[0] for a in arcs} | {a[1] for a in arcs})


def indexed(corpus, structure, sil_prob=None):
    """
    `built`, with its arcs indexed for the search, memoised.

    Indexing mini-librispeech's 1.5 million arcs costs more than building
    them, and four of these tests want the same index -- which is what
    took tier 1 from instant to thirteen seconds on the first draft, the
    same trap `test_prepare_lang.py` records from memoising too late.
    """
    arcs, final = built(corpus, structure, sil_prob)
    settings = dict(SETTINGS[corpus])
    if sil_prob is not None:
        settings["sil_prob"] = sil_prob
    key = (corpus, structure, settings["sil_prob"])
    if key not in _INDEXED:
        _INDEXED[key] = index(arcs)
    return _INDEXED[key], final


def index(arcs):
    """Arcs by source state and input label, for the search below."""
    by = defaultdict(list)
    for source, destination, ilabel, olabel, weight in arcs:
        by[(source, ilabel)].append((destination, olabel, weight))
    return by


def best_path(by, final, sequence, start=0):
    """
    The least-weight path reading `sequence`, and the labels it emits.

    A Dijkstra over (state, position), which is the same shape as
    `BenNevis.decode.graph.weight_of_label_sequence` and for the same
    reason: both structures are **non-deterministic** on the input side
    -- the flower because a pronunciation can be a prefix of another, the
    tree because a one-phone word's final arc leaves the loop beside the
    shared arc for longer words -- so walking greedily finds the wrong
    path or none. Epsilon arcs advance the state and not the position,
    which is what keeps the search finite.

    Returns `(weight, emitted)` or None if the sequence is not accepted.
    """
    seen = {}
    heap = [(0.0, start, 0, ())]
    while heap:
        cost, state, position, emitted = heapq.heappop(heap)
        if position == len(sequence) and state == final:
            return round(cost, 6), emitted
        key = (state, position)
        if seen.get(key, float("inf")) <= cost:
            continue
        seen[key] = cost
        for destination, _olabel, weight in by.get((state, 0), ()):
            heapq.heappush(heap, (cost + weight, destination, position, emitted))
        if position < len(sequence):
            for destination, olabel, weight in by.get((state, sequence[position]), ()):
                heapq.heappush(
                    heap,
                    (
                        cost + weight,
                        destination,
                        position + 1,
                        emitted + ((olabel,) if olabel else ()),
                    ),
                )
    return None


def test_there_are_corpora_to_check():
    assert corpora(), f"no capture under {REFERENCE} has a dictionary and tables"


@pytest.mark.parametrize("corpus", corpora())
class TestTheTreeIsSameFunction:
    """
    The gate that replaces byte-exactness.

    **Built from a subset of the lexicon, deliberately.** Equivalence is
    a per-entry property, so it does not need 206,510 of them -- and
    indexing mini-librispeech's 1.5 million arcs costs a second and
    searching it two more, which is how the first draft took tier 1 from
    instant to thirteen seconds. The subset is chosen to *contain* the
    awkward shapes rather than to be random: every entry that shares a
    pronunciation with a sampled one, and every entry that is a prefix
    of or extends a sampled one, comes along with it. The structural
    claims below still use the whole lexicon, which is one cheap build.
    """

    @staticmethod
    def subset(corpus, seed=0, size=120):
        """
        A sample, closed over the shapes that make this interesting.

        A random slice would mostly miss homophones and prefix pairs,
        which are precisely where sharing a prefix could go wrong.
        """
        lexicon = read_lexicon(REFERENCE / corpus / "dict" / "lexicon.txt")
        if len(lexicon) <= size:
            return lexicon
        chosen = random.Random(seed).sample(lexicon, size)
        pronunciations = {pronunciation for _word, pronunciation in chosen}
        prefixes = {
            pronunciation[:cut]
            for pronunciation in pronunciations
            for cut in range(1, len(pronunciation))
        }
        # One pass, membership only. Comparing every entry against every
        # sampled pronunciation is 206,510 x 120 and took two seconds;
        # asking instead whether any of an entry's *own* prefixes was
        # sampled is linear in its length.
        related = [
            entry
            for entry in lexicon
            if entry[1] in pronunciations
            or entry[1] in prefixes
            or any(
                entry[1][:cut] in pronunciations for cut in range(1, len(entry[1]) + 1)
            )
        ]
        # Order preserved, duplicates removed: the state numbering is a
        # function of the entry order, so a set would make the two
        # structures incomparable between runs.
        seen = set()
        out = []
        for entry in chosen + related:
            if entry not in seen:
                seen.add(entry)
                out.append(entry)
        return out

    @pytest.mark.parametrize("sil_prob", [0.0, 0.5])
    def test_every_pronunciation_maps_to_the_same_word_at_the_same_weight(
        self, corpus, sil_prob
    ):
        """
        An FST's encoding is arbitrary and its function is not, so this
        is the standard CLAUDE.md sets for a replacement graph -- applied
        between two of our own builders, because the flower *is* Kaldi's
        and is kept for that.

        Both silence probabilities, because at zero every weight is zero
        and a comparison of zeros would pass for any structure at all.
        """
        lexicon = self.subset(corpus)
        words = table(corpus, "words.txt")
        phones = table(corpus, "phones.txt")
        position_dependent_phones = SETTINGS[corpus]["position_dependent_phones"]

        sides = {}
        for structure in ("flower", "tree"):
            arcs, final = build_lexicon_arcs(
                lexicon,
                words,
                phones,
                sil_prob=sil_prob,
                sil_phone=silence_phone(corpus),
                position_dependent_phones=position_dependent_phones,
                structure=structure,
            )
            sides[structure] = (index(arcs), final)

        # Which words each pronunciation may legitimately produce.
        # **A single best path's word is not a well-defined thing to
        # compare**, and that is a property of the lexicon rather than of
        # either structure: 50,129 of mini-librispeech's entries share a
        # pronunciation with another word, and at `sil_prob=0` every path
        # costs zero, so "the" best path is whichever the search reaches
        # first. The first version of this test compared them anyway and
        # failed on JAQUELIN, where the two structures each picked a
        # different homophone -- a real ambiguity reported as a defect.
        # What is well defined is the weight, and that the word emitted
        # is one the lexicon gives that pronunciation.
        homophones = {}
        for word, pronunciation in lexicon:
            spelt = (
                position_dependent(pronunciation)
                if position_dependent_phones
                else pronunciation
            )
            homophones.setdefault(spelt, set()).add(words[word])

        compared = 0
        for word, pronunciation in lexicon:
            spelt = (
                position_dependent(pronunciation)
                if position_dependent_phones
                else pronunciation
            )
            sequence = [phones[phone] for phone in spelt]
            mine = best_path(*sides["tree"], sequence)
            theirs = best_path(*sides["flower"], sequence)
            assert theirs is not None, f"the flower rejects {word}: {spelt}"
            assert mine is not None, f"the tree rejects {word}: {spelt}"
            assert (
                mine[0] == theirs[0]
            ), f"{word}: tree weight {mine[0]} against flower {theirs[0]}"
            allowed = homophones[spelt]
            assert set(mine[1]) <= allowed, (
                f"{word}: the tree emits {mine[1]}, which the lexicon does "
                f"not give {spelt}"
            )
            assert set(theirs[1]) <= allowed, f"{word}: the flower emits {theirs[1]}"
            # Whether a word is emitted at all must agree, but it is not
            # always *yes*: with optional silence, a word whose
            # pronunciation **is** the silence phone is read more cheaply
            # by the silence arc than by its own entry. yesno's lexicon
            # has exactly that -- `<SIL> <SIL>` -- so the best path reads
            # the phone and emits nothing, at -log(0.5) against the
            # entry's two -log(0.5)s. Both structures agree, which is the
            # claim; asserting a word was emitted is a claim about the
            # lexicon. yesno passes `--sil_prob 0.0`, so this only
            # arises because the test overrides it to exercise weights.
            assert bool(mine[1]) == bool(theirs[1]), (
                f"{word}: the tree emits {mine[1]} and the flower "
                f"{theirs[1]} -- they disagree about emitting anything"
            )
            compared += 1
        # A comparison over nothing passes, which is how the `G.fst` gate
        # in #41 came to be a test that could not fail.
        assert compared >= min(len(lexicon), 100), compared

    def test_the_subset_contains_the_shapes_it_is_meant_to(self, corpus):
        """
        Otherwise the test above is a random slice and the homophone and
        prefix cases -- the only places sharing could go wrong -- would
        be absent by luck rather than present by construction.
        """
        lexicon = self.subset(corpus)
        counts = {}
        for _word, pronunciation in lexicon:
            counts[pronunciation] = counts.get(pronunciation, 0) + 1
        shared = sum(1 for n in counts.values() if n > 1)
        pronunciations = set(counts)
        prefix_related = sum(
            1
            for pronunciation in pronunciations
            if any(
                other != pronunciation and other[: len(pronunciation)] == pronunciation
                for other in pronunciations
            )
        )
        if corpus == "mini_librispeech":
            assert shared > 0, "no homophones in the subset"
            assert prefix_related > 0, "no prefix pairs in the subset"


@pytest.mark.parametrize("corpus", corpora())
class TestTheTreeIsSmaller:
    def test_no_state_has_a_flowers_out_degree(self, corpus):
        """
        The point of the exercise. On mini-librispeech the flower's loop
        state carries 206,510 arcs and the tree's worst state carries
        195; the decoder's cost is driven by the arcs leaving the states
        it reaches, not by the graph's total size.
        """
        from collections import Counter

        flower_arcs, _ = built(corpus, "flower")
        tree_arcs, _ = built(corpus, "tree")
        flower_worst = max(Counter(a[0] for a in flower_arcs).values())
        tree_worst = max(Counter(a[0] for a in tree_arcs).values())
        # Never worse, everywhere. Not strictly better everywhere: yesno
        # has four words over two phones and nothing to share, so a
        # strict assertion would be a claim about the corpus rather than
        # about the structure.
        assert tree_worst <= flower_worst
        if corpus == "mini_librispeech":
            assert flower_worst > 200_000, flower_worst
            assert tree_worst < 500, tree_worst

    def test_it_has_fewer_arcs_and_states(self, corpus):
        flower_arcs, _ = built(corpus, "flower")
        tree_arcs, _ = built(corpus, "tree")
        assert len(tree_arcs) <= len(flower_arcs)
        assert states_of(tree_arcs) <= states_of(flower_arcs)


class TestTheMeasuredFigures:
    """
    The numbers, written out, so a change of shape shows as a number
    rather than as a ratio nobody checks.

    Not parametrised over corpora: these are mini-librispeech's, and
    yesno has four words over two phones with nothing to share, so it
    has no interesting figure to assert.
    """

    def test_the_mini_librispeech_figures_are_what_was_measured(self):
        """
        Asserted at the recipe's **own** silence probability of 0.5,
        which costs nothing because the tests above already build that
        pair -- where asserting the 0.0 figures too meant two more
        builds of a 206,510-entry lexicon and a second on tier 1.

        At 0.0 the same structures are 1,333,084 arcs over 1,126,575
        states for the flower against 469,947 over 263,438 for the tree:
        **2.84x the arcs and 4.28x the states**, and those are the
        figures worth quoting because they are not clouded by the silence
        split. 0.5 adds one arc per entry to each side -- the split
        between paying for silence and not -- plus three for the silence
        structure itself, `0 -> loop`, `0 -> silence` and
        `silence -> loop`, which exist once however large the lexicon is.
        Leaving those three out is what made the first version of this
        miss by exactly three.
        """
        if "mini_librispeech" not in corpora():
            pytest.skip("no mini-librispeech capture")
        entries = 206_510
        structure_arcs = 3
        flower, _ = built("mini_librispeech", "flower")
        tree, _ = built("mini_librispeech", "tree")

        assert len(flower) == 1_333_084 + entries + structure_arcs
        assert len(tree) == 469_947 + entries + structure_arcs
        assert states_of(flower) == 1_126_575 + 2
        assert states_of(tree) == 263_438 + 2


class TestWhereTheWordRides:
    @staticmethod
    def build(structure, sil_prob=0.0):
        lexicon = [("CAT", ("K", "AE", "T")), ("CAB", ("K", "AE", "B"))]
        words = {"<eps>": 0, "CAT": 1, "CAB": 2}
        phones = {"<eps>": 0, "K": 1, "AE": 2, "T": 3, "B": 4, "SIL": 5}
        return build_lexicon_arcs(
            lexicon, words, phones, sil_prob=sil_prob, structure=structure
        )

    def test_the_flower_puts_it_on_the_first_arc(self):
        arcs, loop = self.build("flower")
        first = [a for a in arcs if a[0] == loop and a[2] == 1]
        assert first and all(a[3] != 0 for a in first), arcs

    def test_the_tree_puts_it_on_the_last(self):
        """
        Forced, not chosen: the first arc is shared between CAT and CAB,
        so it cannot carry either word.
        """
        arcs, loop = self.build("tree")
        leaving = [a for a in arcs if a[0] == loop]
        assert len(leaving) == 1, leaving
        assert leaving[0][3] == 0, "a shared arc carries a word"
        finals = [a for a in arcs if a[1] == loop and a[3] != 0]
        assert {a[3] for a in finals} == {1, 2}

    def test_the_shared_prefix_is_one_path(self):
        arcs, _loop = self.build("tree")
        # K and AE once each, then T and B.
        assert sum(1 for a in arcs if a[2] == 1) == 1
        assert sum(1 for a in arcs if a[2] == 2) == 1
        assert sum(1 for a in arcs if a[2] in (3, 4)) == 2


class TestTheAwkwardCases:
    @staticmethod
    def arcs_for(lexicon, structure, sil_prob=0.0):
        words = {"<eps>": 0}
        for index, (word, _p) in enumerate(lexicon, start=1):
            words.setdefault(word, index)
        phones = {"<eps>": 0, "SIL": 9}
        for _w, pronunciation in lexicon:
            for phone in pronunciation:
                phones.setdefault(phone, len(phones))
        return (
            build_lexicon_arcs(
                lexicon, words, phones, sil_prob=sil_prob, structure=structure
            ),
            words,
            phones,
        )

    def test_homophones_keep_their_own_word(self):
        """
        Two words with one pronunciation share the whole path and differ
        only on the final arc, which is where the word is. 50,129 of
        mini-librispeech's entries are in this case.
        """
        lexicon = [("EWE", ("Y", "UW")), ("YEW", ("Y", "UW"))]
        (arcs, loop), words, phones = self.arcs_for(lexicon, "tree")
        by = index(arcs)
        # Both readings exist and a least-weight search would return
        # only one, so the two-step walk is deliberate: what matters is
        # that *both* words survive the sharing.
        emitted = set()
        for destination, _olabel, _weight in by[(loop, phones["Y"])]:
            for _next, olabel, _w in by[(destination, phones["UW"])]:
                emitted.add(olabel)
        assert emitted == {words["EWE"], words["YEW"]}
        # And the shared prefix really is shared: one arc on Y.
        assert sum(1 for a in arcs if a[2] == phones["Y"]) == 1

    def test_a_pronunciation_that_is_a_prefix_of_another_still_works(self):
        """
        `A` = AH and `ABOUT` = AH B AW T. In the tree, the node after AH
        is both a place a word ends and a place a longer word continues
        through -- which is exactly what the disambiguation symbols
        exist for, and why `add_disambiguation` runs on the entries
        rather than on the FST.
        """
        lexicon = [("A", ("AH",)), ("ABOUT", ("AH", "B", "AW", "T"))]
        (arcs, loop), words, phones = self.arcs_for(lexicon, "tree")
        by = index(arcs)
        short = best_path(by, loop, [phones["AH"]])
        long = best_path(
            by, loop, [phones["AH"], phones["B"], phones["AW"], phones["T"]]
        )
        assert short is not None and set(short[1]) == {words["A"]}
        assert long is not None and set(long[1]) == {words["ABOUT"]}

    def test_a_one_phone_word_leaves_the_loop_directly(self):
        """
        Its final arc *is* its first, so it cannot be shared -- which is
        why the loop's out-degree is the distinct first phones plus the
        one-phone entries, 195 rather than 67 on mini-librispeech.
        """
        lexicon = [("A", ("AH",)), ("B", ("AH", "B"))]
        (arcs, loop), words, _phones = self.arcs_for(lexicon, "tree")
        leaving = [a for a in arcs if a[0] == loop]
        assert len(leaving) == 2, leaving
        assert sorted(a[3] for a in leaving) == [0, words["A"]]

    def test_the_empty_pronunciation_is_accepted(self):
        lexicon = [("<eps-word>", ())]
        (arcs, loop), words, _phones = self.arcs_for(lexicon, "tree")
        assert any(a[3] == words["<eps-word>"] for a in arcs)


class TestItRefusesNonsense:
    def test_an_unknown_structure_is_refused(self):
        with pytest.raises(ValueError, match="flower' or 'tree'"):
            build_lexicon_arcs(
                [("A", ("AH",))],
                {"<eps>": 0, "A": 1},
                {"<eps>": 0, "AH": 1, "SIL": 2},
                structure="trie",
            )


class TestTheSelfLoopCost:
    """
    What `add_self_loops` costs the tree, which is not obvious.

    `AddSelfLoops` in `fstext/pre-determinize-inl.h` loops every final
    state **and every state with a non-epsilon output label on an
    outgoing arc**, with weight One. In the flower every word rides an
    arc leaving the loop state, so exactly one state qualifies and
    `L_disambig` gains a single `#0` self-loop -- which is why
    mini-librispeech's 206,510 entries add "exactly one arc,
    `1 1 347 200004`", as CLAUDE.md records.

    In the tree the words ride the *final* arcs, which leave the
    penultimate nodes, so one self-loop is added **per word-emitting
    state**. That is the part of the saving that the lexicon arc count
    alone does not show, and it is worth a number rather than a
    discovery during a five-hour run.
    """

    @staticmethod
    def with_loops(corpus, structure):
        from BenNevis.lm.lexicon import add_disambiguation

        lexicon = read_lexicon(REFERENCE / corpus / "dict" / "lexicon.txt")
        settings = SETTINGS[corpus]
        if settings["position_dependent_phones"]:
            lexicon = [(word, position_dependent(p)) for word, p in lexicon]
        disambiguated, highest = add_disambiguation(lexicon)
        sil_prob = settings["sil_prob"]
        arcs, final = build_lexicon_arcs(
            disambiguated,
            table(corpus, "words.txt"),
            table(corpus, "phones.txt"),
            sil_prob,
            silence_phone(corpus),
            f"#{highest + 1}" if sil_prob else None,
            structure=structure,
        )
        emitting = {arc[0] for arc in arcs if arc[3] != 0}
        return len(arcs), len(emitting | {final})

    def test_the_flower_needs_exactly_one(self):
        """
        Every word leaves the loop state, which is also the final state,
        so one state qualifies -- the "exactly one arc" CLAUDE.md
        records for mini-librispeech's 206,510 entries.
        """
        if "mini_librispeech" not in corpora():
            pytest.skip("no mini-librispeech capture")
        _arcs, loops = self.with_loops("mini_librispeech", "flower")
        assert loops == 1

    def test_the_tree_needs_one_per_word_emitting_state(self):
        if "mini_librispeech" not in corpora():
            pytest.skip("no mini-librispeech capture")
        _arcs, loops = self.with_loops("mini_librispeech", "tree")
        assert loops == 154_371

    def test_the_saving_survives_them_and_is_smaller_than_it_looks(self):
        """
        The honest figure. 2.28x on the lexicon's own arcs becomes 1.87x
        once the self-loops are counted -- and the out-degree win is
        untouched, because a self-loop adds one arc to a state that had
        two or three rather than to the state that had 206,510.
        """
        if "mini_librispeech" not in corpora():
            pytest.skip("no mini-librispeech capture")
        flower_arcs, flower_loops = self.with_loops("mini_librispeech", "flower")
        tree_arcs, tree_loops = self.with_loops("mini_librispeech", "tree")

        bare = flower_arcs / tree_arcs
        with_loops = (flower_arcs + flower_loops) / (tree_arcs + tree_loops)
        assert bare == pytest.approx(2.28, abs=0.02), bare
        assert with_loops == pytest.approx(1.87, abs=0.02), with_loops
        assert with_loops < bare, "the self-loops must cost something"
        assert with_loops > 1.5, "but the saving must survive them"
