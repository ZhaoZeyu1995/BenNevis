# LibriSpeech recipe with sentencepiece (BPE or unigram) modelling units

This recipe is based on the LibriSpeech ASR corpus.
The models trained with this recipe apply [sentencepiece](https://github.com/google/sentencepiece) (BPE or unigram) modelling units.
For recipes with other modelling units, please refer to other sub-directories in `egs/librispeech`.

Note that one basic requirement for the recipe is to have the `sentencepiece` package installed,
and make sure that the `spm_train` and `spm_encode` commands are available in your `$PATH`.
You may refer to the [sentencepiece installlation guide](https://github.com/google/sentencepiece?tab=readme-ov-file#installation) for more information.
