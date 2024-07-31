/// Converts sample index to frame index based on Conv1d and MaxPool1d operations.
pub fn sample2frame(x: f64) -> f64 {
    // Conv1d & MaxPool1d & SincNet:
    //   * https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    //   * https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    //   * https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/blocks/sincnet.py#L50-L71
    //            kernel_size  stride
    // Conv1d             251      10
    // MaxPool1d            3       3
    // Conv1d               5       1
    // MaxPool1d            3       3
    // Conv1d               5       1
    // MaxPool1d            3       3
    // (L_{in} - 721) / 270 = L_{out}
    (x - 721.0) / 270.0
}

/// Converts frame index to sample index.
pub fn frame2sample(x: f64) -> f64 {
    (x * 270.0) + 721.0
}
