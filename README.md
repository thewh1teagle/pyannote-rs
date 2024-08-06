# pyannote-rs

```console
wget https://github.com/pengzhendong/pyannote-onnx/raw/master/pyannote_onnx/segmentation-3.0.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/5_speakers.wav
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx
cargo run 5_speakers.wav
```

Based on [pyannote-onnx](https://github.com/pengzhendong/pyannote-onnx) ([pyannote-onnx/issues/7](https://github.com/pengzhendong/pyannote-onnx/issues/7))

Future: add speaker embedding with [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
