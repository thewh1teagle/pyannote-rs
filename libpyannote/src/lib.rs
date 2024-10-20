#![allow(non_snake_case)]

use pyannote_rs::Segment;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Wav {
    pub samples: *const i16,
    pub n: usize, // Number of samples
    pub sample_rate: u32,
}

#[repr(C)]
pub struct SegmentResult {
    pub segments: *const Segment, // Pointer to the array of Segment structs
    pub length: usize,            // Number of segments
}

#[no_mangle]
extern "C" fn ReadWave(path: *const c_char) -> *mut Wav {
    let c_str = unsafe {
        assert!(!path.is_null());
        CStr::from_ptr(path)
    };

    let path_str = c_str.to_str().unwrap();
    let (samples, sample_rate) = pyannote_rs::read_wav(path_str).unwrap();
    let samples_count = samples.len();

    let wav = Wav {
        samples: samples.as_ptr(), // Raw pointer to the samples
        n: samples_count,
        sample_rate,
    };
    let boxed_samples = Box::new(samples);
    std::mem::forget(boxed_samples);
    Box::into_raw(Box::new(wav))
}

#[no_mangle]
extern "C" fn GetSegments(wav: *const Wav, model_path: *const c_char) -> *mut SegmentResult {
    if wav.is_null() || model_path.is_null() {
        return ptr::null_mut();
    }

    let wav = unsafe { &*wav };

    let c_str = unsafe { CStr::from_ptr(model_path) };
    let model_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let samples: &[i16] = unsafe { std::slice::from_raw_parts(wav.samples, wav.n) };

    match pyannote_rs::get_segments(samples, wav.sample_rate, model_str) {
        Ok(segments) => {
            let segments_vec: Vec<Segment> = segments.flatten().collect();
            let length = segments_vec.len();

            if length == 0 {
                return ptr::null_mut();
            }

            let segment_result = SegmentResult {
                segments: segments_vec.as_ptr(),
                length,
            };

            let boxed_segments = Box::new(segments_vec);
            std::mem::forget(boxed_segments); // Prevent Rust from deallocating the segments

            Box::into_raw(Box::new(segment_result))
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
extern "C" fn GetSegmentAt(segment_result: *const SegmentResult, index: usize) -> *const Segment {
    if segment_result.is_null() {
        return ptr::null();
    }

    let segment_result = unsafe { &*segment_result };

    if index >= segment_result.length {
        return ptr::null();
    }

    unsafe { segment_result.segments.add(index) }
}

#[no_mangle]
extern "C" fn FreeWav(wav: *mut Wav) {
    if !wav.is_null() {
        unsafe {
            // The wav is freed, but we need to ensure that we also drop the samples
            let wav = Box::from_raw(wav);
            let samples = std::slice::from_raw_parts_mut(wav.samples as *mut i16, wav.n);
            let _ = Box::from_raw(samples);
        }
    }
}

#[no_mangle]
extern "C" fn FreeSegmentResult(segment_result: *mut SegmentResult) {
    if !segment_result.is_null() {
        unsafe {
            let _ = Box::from_raw(segment_result); // Free the SegmentResult struct
        }
    }
}
