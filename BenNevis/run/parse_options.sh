# Read a script's own long options, without Kaldi.
#
# A replacement for `utils/parse_options.sh`, which is a symlink into the
# Kaldi checkout -- so **a recipe could not parse `--stage` without
# Kaldi**, long after every stage `--stage` selects had become Python.
# That is the same defect `prepare_topo.sh` had and the same fix: the
# script needed a Kaldi checkout for an argument parser.
#
# Sourced, not run, because it sets the caller's variables:
#
#     stage=0
#     stop_stage=100
#     . ./run/parse_options.sh || exit 1
#
# Kaldi's semantics are reproduced, because seven recipes and four
# `run/*.sh` are written against them:
#
# * `--name value` sets `name`, with dashes in the option becoming
#   underscores -- so `--stop-stage` and `--stop_stage` both set
#   `stop_stage`;
# * **the variable must already exist**, and an unknown option is an
#   error rather than being ignored. That is the whole value of the
#   thing: `--stop_stge 4` would otherwise run every stage;
# * `--name=value` is refused, because Kaldi refuses it and a recipe
#   written for one and run under the other would silently take a
#   default;
# * a variable whose current value is `true` or `false` may only be set
#   to `true` or `false`, which catches `--write-utt2num-frames yes`;
# * parsing stops at the first argument that is not an option, leaving
#   the positional arguments in `$@`.
#
# `--help` is handled only if the caller has set `help_message`, as
# Kaldi's does.
#
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2026

while true; do
    case "${1:-}" in
    "")
        break
        ;;
    --help|-h)
        if [ -z "${help_message:-}" ]; then
            echo "$0: no help message is set" >&2
        else
            printf "%s\n" "$help_message" >&2
        fi
        exit 0
        ;;
    --*=*)
        echo "$0: --name=value is not supported; use --name value" >&2
        echo "$0: got '$1'" >&2
        exit 1
        ;;
    --*)
        name=$(echo "$1" | sed 's/^--//; s/-/_/g')
        # `${name+set}` is empty only when the variable has never been
        # assigned, so a declared-but-empty option is still settable.
        if ! eval "[ -n \"\${$name+set}\" ]"; then
            echo "$0: unknown option '$1'" >&2
            exit 1
        fi
        if [ $# -lt 2 ]; then
            echo "$0: option '$1' needs a value" >&2
            exit 1
        fi
        old=$(eval "printf '%s' \"\$$name\"")
        case "$old" in
        true|false)
            case "$2" in
            true|false) ;;
            *)
                echo "$0: '$1' is a boolean and was given '$2'" >&2
                exit 1
                ;;
            esac
            ;;
        esac
        eval "$name=\$2"
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

true
