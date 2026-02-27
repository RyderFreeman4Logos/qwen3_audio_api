use crate::config::ResponseFormat;
use crate::error::ApiError;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;
use std::process::{Command, Stdio};

/// Encode f32 audio samples to the requested format.
pub fn encode_audio(
    samples: &[f32],
    sample_rate: u32,
    format: ResponseFormat,
) -> Result<Vec<u8>, ApiError> {
    match format {
        ResponseFormat::Wav => encode_wav(samples, sample_rate),
        ResponseFormat::Pcm => Ok(encode_pcm(samples)),
        ResponseFormat::Flac | ResponseFormat::Mp3 | ResponseFormat::Opus | ResponseFormat::Aac => {
            encode_with_ffmpeg(samples, sample_rate, format)
        }
    }
}

/// Encode as 16-bit PCM WAV (mono).
fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, ApiError> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut buffer, spec)
            .map_err(|e| ApiError::internal(format!("WAV encode error: {e}")))?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_val = (clamped * 32767.0) as i16;
            writer
                .write_sample(i16_val)
                .map_err(|e| ApiError::internal(format!("WAV write error: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| ApiError::internal(format!("WAV finalize error: {e}")))?;
    }
    Ok(buffer.into_inner())
}

/// Encode as raw 16-bit signed little-endian PCM.
fn encode_pcm(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&i16_val.to_le_bytes());
    }
    bytes
}

/// Encode via ffmpeg for mp3/opus/aac/flac.
fn encode_with_ffmpeg(
    samples: &[f32],
    sample_rate: u32,
    format: ResponseFormat,
) -> Result<Vec<u8>, ApiError> {
    let wav_bytes = encode_wav(samples, sample_rate)?;

    let (codec, container) = match format {
        ResponseFormat::Mp3 => ("libmp3lame", "mp3"),
        ResponseFormat::Opus => ("libopus", "ogg"),
        ResponseFormat::Aac => ("aac", "adts"),
        ResponseFormat::Flac => ("flac", "flac"),
        _ => unreachable!(),
    };

    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-acodec",
            codec,
            "-f",
            container,
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ApiError::internal(format!("Failed to run ffmpeg: {e}")))?;

    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(&wav_bytes)
            .map_err(|e| ApiError::internal(format!("Failed to write to ffmpeg stdin: {e}")))?;
    }
    // Drop stdin to signal EOF
    child.stdin.take();

    let output = child
        .wait_with_output()
        .map_err(|e| ApiError::internal(format!("ffmpeg error: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ApiError::internal(format!("ffmpeg encoding failed: {stderr}")));
    }

    Ok(output.stdout)
}

/// Apply speed adjustment by resampling via linear interpolation.
/// Speed > 1.0 = faster (shorter audio), speed < 1.0 = slower (longer audio).
pub fn apply_speed(samples: &[f32], speed: f32) -> Vec<f32> {
    if (speed - 1.0).abs() < f32::EPSILON || samples.is_empty() {
        return samples.to_vec();
    }

    let new_length = (samples.len() as f64 / speed as f64) as usize;
    if new_length == 0 {
        return samples.to_vec();
    }

    let mut output = Vec::with_capacity(new_length);
    for i in 0..new_length {
        let src_pos = i as f64 * speed as f64;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        if idx + 1 < samples.len() {
            output.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }
    output
}
