MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$MAIN_ROOT/BenNevis/bin:$MAIN_ROOT/BenNevis/bin/topos:$KALDI_ROOT/tools/openfst/bin/:$KALDI_ROOT/tools/sctk/bin/:$KALDI_ROOT/egs/wsj/s5/utils:$PATH
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PYTHONPATH=$MAIN_ROOT:$PYTHONPATH

export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
