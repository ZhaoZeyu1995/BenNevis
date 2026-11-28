"""
Keeping a Kaldi data directory consistent, in Python.

`utils/fix_data_dir.sh` is the one stage-2 script that is not arithmetic:
it makes sure the several files describing a corpus agree about which
utterances exist, drops the ones that do not appear everywhere, and sorts
what it leaves. Feature extraction is what makes that necessary -- an
utterance whose audio could not be read has no row in `feats.scp` while
still having a transcription, and training on the difference is how yesno
came to claim a 31-utterance test set that was really 29.

**This is a reimplementation of Kaldi's script read off the script**, not
an approximation of what it seems to do, and two of its rules are not
guessable:

**A speaker with no CMVN statistics is dropped, together with all of its
utterances.** `filter_speakers` calls its own helper with the arguments
the other way round -- `filter_file $f $tmpdir/speakers`, filtering the
*speaker list* by `cmvn.scp` rather than the other way about -- and then
regenerates `utt2spk` from the surviving `spk2utt`. So the direction of
that one call decides whether a speaker whose statistics failed loses its
utterances or keeps them, and it loses them.

**`wav.scp` is utterance-indexed only when there is no `segments` file.**
With segments it is indexed by *recording*, and filtering it against an
utterance list would empty it. Neither recipe here has segments, and this
module **refuses** a directory that does rather than doing the wrong
thing quietly -- the same choice `read_fbank_conf` makes for an option it
does not understand.

Sorting is by **bytes**, because Kaldi's script exports `LC_ALL=C`. That
is the same trap as yesno's `README~`: under a locale where `R` sorts
before a digit, the split moves.

Functions
---------
read_table
    One of these files as a list of key-and-rest pairs.
write_table
    The inverse, with a trailing newline per line.
sort_unique
    Byte-sorted by key, one line per key.
fix
    The whole of `fix_data_dir.sh` for the files a recipe here has.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

__all__ = [
    "UTTERANCE_FILES",
    "INTERSECT_FILES",
    "POSITIVE_FILES",
    "SPEAKER_FILES",
    "UNSUPPORTED_FILES",
    "DataDirError",
    "read_table",
    "write_table",
    "sort_unique",
    "fix",
]

#: The utterance-indexed files this rewrites. `wav.scp` is here because
#: neither recipe has a `segments` file; see the module docstring.
UTTERANCE_FILES = (
    "utt2spk",
    "utt2uniq",
    "feats.scp",
    "vad.scp",
    "text",
    "wav.scp",
    "utt2lang",
    "utt2num_frames",
    "utt2dur",
    "feats.cmvn.scp",
)

#: The subset whose keys *narrow* the surviving utterance list. Kaldi
#: separates these two lists and the difference matters: `vad.scp` is
#: rewritten but never intersected, so a stale one cannot delete
#: utterances. `feats.cmvn.scp` is ours and is the same case -- it is
#: written by a later stage, so on a re-run an old copy is behind the
#: others and must not be allowed to narrow them.
INTERSECT_FILES = (
    "feats.scp",
    "text",
    "wav.scp",
    "utt2lang",
    "utt2num_frames",
    "utt2dur",
)

#: The files whose value must be a positive number for the utterance to
#: survive, which is Kaldi's `awk '{ if (NF == 2 && $2 > 0) { print }}'`
#: over `utt2dur` and `utt2num_frames`. A zero-length utterance is not
#: something to train on, and keeping it at zero would reach the collate
#: function as an empty matrix -- or, for `utt2dur`, divide by zero in the
#: duration samplers.
POSITIVE_FILES = ("utt2num_frames", "utt2dur")

#: The speaker-indexed files. `spk2utt` is derived from `utt2spk` rather
#: than filtered, which is Kaldi's own rule: "throughout this program, we
#: regard utt2spk as primary and spk2utt as derived".
#:
#: **`spk2gender` is here because leaving it out broke mini-librispeech.**
#: `local/data_prep.sh` writes it from `SPEAKERS.TXT` -- one entry per
#: chapter-speaker -- and nothing in BenNevis reads it, so it was listed
#: as unsupported and `fix` refused the whole directory. Kaldi's
#: `filter_speakers` treats it exactly like `cmvn.scp`, in both
#: directions, so that is what this does.
SPEAKER_FILES = ("cmvn.scp", "spk2gender")

#: Files Kaldi's script handles and this does not. Present in a
#: directory, each would be left unfiltered while everything around it
#: was filtered -- worse than refusing, because `segments` makes `wav.scp`
#: **recording**-indexed and the two `reco2*` files are indexed that way
#: too.
#:
#: **This list used to be longer, and that cost a five-and-a-half-hour
#: slot.** `spk2gender`, `utt2uniq`, `vad.scp` and `utt2lang` were on it
#: because nothing in BenNevis reads them -- but `local/data_prep.sh`
#: *writes* `spk2gender`, so `fix` refused every mini-librispeech data
#: directory and stage 2 failed on every run from the stage-2 conversion
#: onward. The four are handled now, the way Kaldi handles them. The
#: lesson is that refusing a file nothing reads is only safe if nothing
#: **writes** it either, and the writers here are Kaldi's own
#: corpus-preparation scripts.
UNSUPPORTED_FILES = (
    "segments",
    "reco2dur",
    "reco2file_and_channel",
    "images.scp",
)


class DataDirError(Exception):
    """A data directory this cannot fix, or must not."""


def read_table(path: Path) -> List[Tuple[str, str]]:
    """
    Read a Kaldi table as a list of key-and-rest pairs.

    Arguments
    ---------
    path: Path
        The file to read. Blank lines are skipped, as `awk` would.

    Returns
    -------
    List[Tuple[str, str]]
        The key and the remainder of each line, in file order. The
        remainder keeps its interior spacing, because a `wav.scp` entry is
        a shell command and reformatting one would change what it runs.

    Notes
    -----
    A line holding a key and nothing else is kept, with an empty
    remainder. That is an **empty transcription**, which `fix_data_dir.sh`
    tolerates -- it is `validate_data_dir.sh` that complains -- so
    refusing it here would reject a data directory Kaldi accepts.
    `write_table` writes such a line back as the bare key.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split(None, 1)
            rows.append((fields[0], fields[1].strip() if len(fields) > 1 else ""))
    return rows


def write_table(path: Path, rows) -> None:
    """
    Write key-and-rest pairs back out, one per line.

    Arguments
    ---------
    path: Path
        Where to write.
    rows: Iterable[Tuple[str, str]]
        The pairs, written in the order given.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for key, rest in rows:
            handle.write(f"{key} {rest}\n" if rest else f"{key}\n")


def sort_unique(rows) -> List[Tuple[str, str]]:
    """
    Sort by key as bytes, keeping one line per key.

    This is `sort -k1,1 -u` under `LC_ALL=C`. Where a key appears twice
    the first after sorting is kept, which is what `sort -u` does for
    lines it considers equal.

    Arguments
    ---------
    rows: Iterable[Tuple[str, str]]
        The pairs to sort.

    Returns
    -------
    List[Tuple[str, str]]
    """
    ordered = sorted(rows, key=lambda row: row[0].encode("utf-8"))
    out = []
    for key, rest in ordered:
        if out and out[-1][0] == key:
            continue
        out.append((key, rest))
    return out


def _positive(text: str) -> bool:
    """
    Whether a table's value is a number above zero.

    Arguments
    ---------
    text: str
        The remainder of the line, which should be one number.

    Returns
    -------
    bool
        False for anything unparseable, which is how `awk` treats a
        malformed line under `$2 > 0`.
    """
    try:
        return float(text) > 0.0
    except ValueError:
        return False


def _keys(rows) -> List[str]:
    return [key for key, _rest in rows]


def fix(
    data: Path,
    *,
    report: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """
    Make a data directory's files agree about which utterances exist.

    The originals are copied into `<data>/.backup` first, exactly as
    Kaldi's script does, so a run that drops more than expected can be
    inspected rather than re-prepared.

    Arguments
    ---------
    data: Path
        The data directory.
    report: Optional[Callable[[str], None]]
        Called with each line of commentary -- what was sorted, what was
        filtered and by how much. `print` or a logger's `info`.

    Returns
    -------
    Dict[str, object]
        `kept` and `dropped` utterance counts, `speakers`, and the list of
        `files` rewritten. Returned rather than only logged so a test can
        assert on what happened instead of on what was said about it.

    Raises
    ------
    DataDirError
        If the directory or `utt2spk` is missing, if a file this does not
        handle is present, or if no utterance survives -- which Kaldi also
        treats as fatal, because an empty data directory trains on nothing
        and scores as a run of deletions.
    """
    say = report if report is not None else (lambda _line: None)

    if not data.is_dir():
        raise DataDirError(f"no such directory {data}")
    utt2spk_path = data / "utt2spk"
    if not utt2spk_path.is_file():
        raise DataDirError(f"no such file {utt2spk_path}")

    present = [name for name in UNSUPPORTED_FILES if (data / name).is_file()]
    if present:
        raise DataDirError(
            f"{data} contains {', '.join(present)}, which this does not "
            f"handle; use Kaldi's utils/fix_data_dir.sh for it. Leaving "
            f"such a file unfiltered while filtering the rest would be "
            f"worse than refusing"
        )

    backup = data / ".backup"
    backup.mkdir(parents=True, exist_ok=True)

    # Step one, Kaldi's `check_sorted` loop: back up, then sort and
    # de-duplicate every table that exists.
    tables: Dict[str, List[Tuple[str, str]]] = {}
    for name in UTTERANCE_FILES + SPEAKER_FILES + ("spk2utt",):
        path = data / name
        if not path.is_file():
            continue
        shutil.copyfile(path, backup / name)
        rows = read_table(path)
        ordered = sort_unique(rows)
        if ordered != rows:
            say(f"{name} was not sorted and unique; sorting it")
        tables[name] = ordered

    # Step two, `filter_speakers`: utt2spk is primary, spk2utt derived,
    # and a speaker absent from cmvn.scp takes its utterances with it.
    speakers = []
    for _utt, speaker in tables["utt2spk"]:
        if speaker not in speakers:
            speakers.append(speaker)
    for name in SPEAKER_FILES:
        if name in tables:
            known = {key for key, _rest in tables[name]}
            missing = [speaker for speaker in speakers if speaker not in known]
            if missing:
                say(
                    f"{len(missing)} of {len(speakers)} speakers are absent "
                    f"from {name}; dropping them and their utterances"
                )
            speakers = [speaker for speaker in speakers if speaker in known]
    if not speakers:
        raise DataDirError(f"{data}: no speaker survived; not proceeding further")
    surviving_speakers = set(speakers)

    # Step three, `filter_utts`: intersect the utterance lists. Kaldi
    # starts from utt2spk's keys and narrows by each table in turn, which
    # is an intersection with the order of utt2spk preserved.
    utterances = [
        utt for utt, speaker in tables["utt2spk"] if speaker in surviving_speakers
    ]
    for name in INTERSECT_FILES:
        if name not in tables:
            continue
        rows = tables[name]
        if name in POSITIVE_FILES:
            # `awk '{ if (NF == 2 && $2 > 0) { print }}'`. Parsed as a
            # float rather than with `isdigit`, because `utt2dur` holds
            # seconds -- `1.5` is not a digit string, and treating it as
            # malformed would drop every utterance and leave a data
            # directory with nothing in it.
            known = {key for key, rest in rows if _positive(rest)}
        else:
            known = {key for key, _rest in rows}
        before = len(utterances)
        utterances = [utt for utt in utterances if utt in known]
        if len(utterances) != before:
            say(f"{before - len(utterances)} utterances are absent from {name}")

    if not utterances:
        raise DataDirError(f"{data}: no utterances remained; not proceeding further")

    original = len(tables["utt2spk"])
    if len(utterances) != original:
        say(f"kept {len(utterances)} utterances out of {original}")
    else:
        say(f"kept all {original} utterances")

    # Step four: rewrite everything to the surviving utterances, then
    # re-derive spk2utt and re-filter the speaker files, which is Kaldi's
    # second `filter_speakers` call.
    keep = set(utterances)
    rewritten = []
    for name in UTTERANCE_FILES:
        if name not in tables:
            continue
        rows = [row for row in tables[name] if row[0] in keep]
        write_table(data / name, rows)
        rewritten.append(name)

    spk2utt = {}
    for utt, speaker in read_table(data / "utt2spk"):
        spk2utt.setdefault(speaker, []).append(utt)
    write_table(
        data / "spk2utt",
        sort_unique((speaker, " ".join(utts)) for speaker, utts in spk2utt.items()),
    )
    rewritten.append("spk2utt")

    for name in SPEAKER_FILES:
        if name not in tables:
            continue
        rows = [row for row in tables[name] if row[0] in spk2utt]
        write_table(data / name, rows)
        rewritten.append(name)

    say(f"old files are kept in {backup}")
    return {
        "kept": len(utterances),
        "dropped": original - len(utterances),
        "speakers": len(spk2utt),
        "files": rewritten,
    }
