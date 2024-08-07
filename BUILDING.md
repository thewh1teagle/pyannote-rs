# Building

### Prerequisites

[Cargo](https://www.rust-lang.org/tools/install) | [Clang](https://releases.llvm.org/download.html) | [Cmake](https://cmake.org/download/)

_Prepare repo (or use cargo add)_

```console
git clone https://github.com/thewh1teagle/pyannote-rs --recursive
```

_Prepare models_

```console
wget https://github.com/pengzhendong/pyannote-onnx/raw/master/pyannote_onnx/segmentation-3.0.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx
wget https://github.com/thewh1teagle/sherpa-rs/releases/download/v0.1.0/6_speakers.wav
```

_Build Example_

```console
cargo run --example diarize 6_speakers.wav
```

_Gotachas_

On `Windows` static linking may fail.
You can resolve it by creating `.cargo/config.toml` next to `Cargo.toml` with the following:

```toml
[target.'cfg(windows)']
rustflags = ["-Ctarget-feature=+crt-static"]
```

Or set the environment variable `RUSTFLAGS` to `-C target-feature=+crt-static`
