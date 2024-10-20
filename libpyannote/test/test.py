"""
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/segmentation-3.0.onnx
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav
cargo build --release
python3 test.py ../../target/release/liblibpyannote.dylib segmentation-3.0.onnx 6_speakers.wav
"""

import sys
import ctypes
from ctypes import c_double, c_size_t, POINTER, c_int16, c_uint32


class Wav(ctypes.Structure):
    _fields_ = [
        ("samples", POINTER(c_int16)),
        ("sample_count", c_size_t),
        ("sample_rate", c_uint32),
    ]


class Segment(ctypes.Structure):
    _fields_ = [
        ("start", c_double),
        ("end", c_double),
        ("samples", POINTER(c_int16)),
        ("samples_count", c_size_t),
    ]


class SegmentResult(ctypes.Structure):
    _fields_ = [
        ("segments", POINTER(Segment)),
        ("length", c_size_t),
    ]


def main():
    lib_path = sys.argv[1]
    model_path = sys.argv[2]
    wav_file_path = sys.argv[3]

    lib = ctypes.CDLL(lib_path)

    # Set the function return types
    lib.ReadWave.restype = POINTER(Wav)
    lib.GetSegments.restype = POINTER(SegmentResult)
    lib.GetSegmentAt.restype = POINTER(Segment)

    try:
        # Read WAV file
        wav = lib.ReadWave(wav_file_path.encode("utf-8"))
        print(
            f"Loaded WAV: {wav.contents.sample_rate} Hz, {wav.contents.sample_count} samples"
        )

        segments = lib.GetSegments(wav, model_path.encode("utf-8"))
        print(f"Segments obtained, total: {segments.contents.length}")

        for i in range(segments.contents.length):
            segment = lib.GetSegmentAt(segments, i)
            print(
                f"Segment {i}: start_time={segment.contents.start}, "
                f"end_time={segment.contents.end}, samples_count={segment.contents.samples_count}"
            )

        # Free the segments and WAV
        lib.FreeSegmentResult(segments)
        lib.FreeWav(wav)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
