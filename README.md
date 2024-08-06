# pyannote-rs

## Install

```console
cargo add pyannote-rs
```

## Usage

See [Building](BUILDING.md)

## Examples

See [examples](examples)

## How it works

pyannote-rs uses 2 models to achieve speaker diarization. The first one is [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) for segmentation (knowing when speech occur)

The second model is [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) which uses to identify who is speaking.

All the inference happens in [onnxruntime](https://onnxruntime.ai/)

The sementation model expects input of at most 10s audio. So we feed it with sliding window of 10s (iterate 10s and feed).

The embedding model expects input of `filter banks` (extracted features from the audio), so we use [knf-rs](https://github.com/thewh1teagle/knf-rs) to extract them.

For speaker comparision (Eg. is Alis spoke again?) we use cosine similarity.

## Credits

Big thanks to [pyannote-onnx](https://github.com/pengzhendong/pyannote-onnx) and [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
