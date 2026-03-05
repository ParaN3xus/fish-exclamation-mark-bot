use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use lewton::inside_ogg::OggStreamReader;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use tracing::{error, info};

use crate::config::{AppConfig, AudioConfig};

#[derive(Debug)]
struct AudioRing {
    data: Vec<f32>,
    write_pos: usize,
    channels: usize,
    total_frames_written: u64,
}

impl AudioRing {
    fn new(capacity_frames: usize, channels: usize) -> Self {
        Self {
            data: vec![0.0; capacity_frames * channels.max(1)],
            write_pos: 0,
            channels: channels.max(1),
            total_frames_written: 0,
        }
    }

    fn channels(&self) -> usize {
        self.channels
    }

    fn total_frames(&self) -> u64 {
        self.total_frames_written
    }

    fn push_samples(&mut self, samples: &[f32]) {
        if self.data.is_empty() {
            return;
        }
        let frames = samples.len() / self.channels.max(1);
        self.total_frames_written = self.total_frames_written.saturating_add(frames as u64);

        for &s in samples {
            self.data[self.write_pos] = s;
            self.write_pos = (self.write_pos + 1) % self.data.len();
        }
    }

    fn latest(&self, frame_count: usize) -> Vec<f32> {
        let total = frame_count * self.channels;
        if total == 0 || total > self.data.len() {
            return Vec::new();
        }

        let start = (self.write_pos + self.data.len() - total) % self.data.len();
        let mut out = Vec::with_capacity(total);
        for i in 0..total {
            out.push(self.data[(start + i) % self.data.len()]);
        }
        out
    }
}

pub struct AudioCapture {
    _stream: cpal::Stream,
    pub sample_rate: u32,
    pub channels: usize,
}

impl AudioCapture {
    fn new(ring: Arc<Mutex<AudioRing>>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no audio output device for loopback")?;
        let supported = device.default_output_config()?;
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.config();
        let channels = config.channels as usize;
        let sample_rate = config.sample_rate;

        let device_desc = device
            .description()
            .map(|d| d.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());

        info!(
            device = %device_desc,
            sample_rate,
            channels,
            format = ?sample_format,
            "audio loopback initialized"
        );

        let err_fn = |err| error!(error = %err, "audio stream error");
        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                let ring = ring.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _| {
                        if let Ok(mut r) = ring.lock() {
                            r.push_samples(data);
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            cpal::SampleFormat::I16 => {
                let ring = ring.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        if let Ok(mut r) = ring.lock() {
                            let mut tmp = Vec::with_capacity(data.len());
                            tmp.extend(data.iter().map(|v| *v as f32 / i16::MAX as f32));
                            r.push_samples(&tmp);
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            cpal::SampleFormat::U16 => {
                let ring = ring.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _| {
                        if let Ok(mut r) = ring.lock() {
                            let mut tmp = Vec::with_capacity(data.len());
                            tmp.extend(
                                data.iter()
                                    .map(|v| (*v as f32 / u16::MAX as f32) * 2.0 - 1.0),
                            );
                            r.push_samples(&tmp);
                        }
                    },
                    err_fn,
                    None,
                )?
            }
            _ => bail!("unsupported sample format for loopback"),
        };

        stream.play()?;
        Ok(Self {
            _stream: stream,
            sample_rate,
            channels,
        })
    }
}

struct FeatureExtractor {
    fft_size: usize,
    hop: usize,
    bar_count: usize,
    window: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl FeatureExtractor {
    fn new(fft_size: usize, hop: usize, bar_count: usize) -> Self {
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (fft_size as f32 - 1.0)).cos())
            })
            .collect();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        Self {
            fft_size,
            hop,
            bar_count,
            window,
            fft,
        }
    }

    fn feature_of_frame(&self, mono: &[f32]) -> Option<Vec<f32>> {
        if mono.len() < self.fft_size {
            return None;
        }

        let mut spectrum: Vec<Complex<f32>> = mono[..self.fft_size]
            .iter()
            .enumerate()
            .map(|(i, &s)| Complex::new(s * self.window[i], 0.0))
            .collect();
        self.fft.process(&mut spectrum);

        let half = self.fft_size / 2;
        let mut feat = vec![0.0f32; self.bar_count];
        for (i, bucket) in feat.iter_mut().enumerate() {
            let t0 = i as f32 / self.bar_count as f32;
            let t1 = (i + 1) as f32 / self.bar_count as f32;
            let bin0 = (t0 * t0 * (half as f32 - 1.0)) as usize;
            let bin1 = ((t1 * t1 * (half as f32 - 1.0)) as usize).max(bin0 + 1);

            let mut max_mag: f32 = 0.0;
            for b in bin0..=bin1.min(half - 1) {
                let re = spectrum[b].re;
                let im = spectrum[b].im;
                let mag = (re * re + im * im).sqrt() / half as f32;
                max_mag = max_mag.max(mag);
            }
            *bucket = (1.0 + max_mag * 200.0).ln();
        }

        let norm = feat.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm < 1e-6 {
            return None;
        }
        for v in &mut feat {
            *v /= norm;
        }
        Some(feat)
    }

    fn sequence_from_samples(&self, mono: &[f32]) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let mut start = 0usize;
        while start + self.fft_size <= mono.len() {
            if let Some(f) = self.feature_of_frame(&mono[start..start + self.fft_size]) {
                out.push(f);
            }
            start += self.hop;
        }
        out
    }
}

struct TemplateMatcher {
    seq: Vec<Vec<f32>>,
    threshold: f32,
    min_energy: f32,
    cooldown_ms: u64,
    cooldown_until_ms: u64,
    suppress_until_window_start_frame: Option<u64>,
    last_similarity: f32,
}

impl TemplateMatcher {
    fn new(seq: Vec<Vec<f32>>, threshold: f32, min_energy: f32, cooldown_ms: u64) -> Self {
        Self {
            seq,
            threshold,
            min_energy,
            cooldown_ms,
            cooldown_until_ms: 0,
            suppress_until_window_start_frame: None,
            last_similarity: 0.0,
        }
    }

    fn score(&mut self, live_seq: &[Vec<f32>]) -> Option<(f32, f32, usize)> {
        let (dist, end_idx) = subseq_dtw_distance(&self.seq, live_seq)?;
        let sim = (-dist).exp();
        self.last_similarity = sim;
        Some((sim, dist, end_idx))
    }

    fn update(
        &mut self,
        live_seq: &[Vec<f32>],
        energy: f32,
        now_ms: u64,
        window_start_frame: u64,
        hop: usize,
        fft_size: usize,
    ) -> bool {
        if let Some(end_frame) = self.suppress_until_window_start_frame {
            if window_start_frame <= end_frame {
                return false;
            }
            self.suppress_until_window_start_frame = None;
        }

        if now_ms < self.cooldown_until_ms || energy < self.min_energy {
            return false;
        }

        if let Some((sim, _dist, end_idx)) = self.score(live_seq) {
            if sim >= self.threshold {
                self.cooldown_until_ms = now_ms + self.cooldown_ms;
                let end_abs = window_start_frame
                    .saturating_add((end_idx.saturating_mul(hop) + fft_size) as u64);
                self.suppress_until_window_start_frame = Some(end_abs);
                return true;
            }
        }

        false
    }
}

pub struct AudioEvents {
    pub bite_hit: bool,
    pub success_hit: bool,
    pub fail_hit: bool,
    pub collected_hit: bool,
    pub bite_similarity: f32,
    pub success_similarity: f32,
    pub fail_similarity: f32,
    pub collected_similarity: f32,
}

pub struct AudioEngine {
    ring: Arc<Mutex<AudioRing>>,
    _capture: AudioCapture,
    extractor: FeatureExtractor,
    live_frames_needed: usize,
    poll_ms: u64,
    hop: usize,
    fft_size: usize,
    loudness_target_rms: f32,
    loudness_gain_min: f32,
    loudness_gain_max: f32,
    last_eval_ms: u64,
    bite_matcher: TemplateMatcher,
    success_matcher: TemplateMatcher,
    fail_matcher: TemplateMatcher,
    collected_matcher: TemplateMatcher,
}

impl AudioEngine {
    pub fn new(app: &AppConfig, exe_dir: &Path) -> Result<Self> {
        let cfg = &app.audio;
        if cfg.hop == 0 {
            bail!("audio.hop must be > 0");
        }

        let ring = Arc::new(Mutex::new(AudioRing::new(
            (48000.0 * cfg.live_seconds) as usize,
            2,
        )));
        let capture = AudioCapture::new(ring.clone())?;
        let target_sr = capture.sample_rate;

        if let Ok(mut lock) = ring.lock() {
            *lock = AudioRing::new((target_sr as f32 * cfg.live_seconds) as usize, capture.channels);
        }

        let extractor = FeatureExtractor::new(cfg.fft_size, cfg.hop, cfg.bar_count);

        let bite_tpl = app.bite_template_path(exe_dir);
        let success_tpl = app.success_template_path(exe_dir);
        let fail_tpl = app.fail_template_path(exe_dir);
        let collected_tpl = app.collected_template_path(exe_dir);
        if !bite_tpl.exists() {
            bail!("bite template not found: {}", bite_tpl.display());
        }
        if !success_tpl.exists() {
            bail!("success template not found: {}", success_tpl.display());
        }
        if !fail_tpl.exists() {
            bail!("fail template not found: {}", fail_tpl.display());
        }
        if !collected_tpl.exists() {
            bail!("collected template not found: {}", collected_tpl.display());
        }

        let bite_dec = decode_audio(&bite_tpl)?;
        let success_dec = decode_audio(&success_tpl)?;
        let fail_dec = decode_audio(&fail_tpl)?;
        let collected_dec = decode_audio(&collected_tpl)?;

        let bite_samples = take_prefix_seconds(
            &loudness_match_cfg(&trim_silence(&resample_linear(
                &bite_dec.mono,
                bite_dec.sample_rate,
                target_sr,
            ), cfg), cfg),
            target_sr,
            cfg.template_prefix_seconds,
        );
        let success_samples = take_prefix_seconds(
            &loudness_match_cfg(&trim_silence(&resample_linear(
                &success_dec.mono,
                success_dec.sample_rate,
                target_sr,
            ), cfg), cfg),
            target_sr,
            cfg.template_prefix_seconds,
        );
        let fail_samples = take_prefix_seconds(
            &loudness_match_cfg(&trim_silence(&resample_linear(
                &fail_dec.mono,
                fail_dec.sample_rate,
                target_sr,
            ), cfg), cfg),
            target_sr,
            cfg.template_prefix_seconds,
        );
        let collected_samples = take_prefix_seconds(
            &loudness_match_cfg(&trim_silence(&resample_linear(
                &collected_dec.mono,
                collected_dec.sample_rate,
                target_sr,
            ), cfg), cfg),
            target_sr,
            cfg.template_prefix_seconds,
        );

        let bite_seq = extractor.sequence_from_samples(&bite_samples);
        let success_seq = extractor.sequence_from_samples(&success_samples);
        let fail_seq = extractor.sequence_from_samples(&fail_samples);
        let collected_seq = extractor.sequence_from_samples(&collected_samples);

        if bite_seq.is_empty()
            || success_seq.is_empty()
            || fail_seq.is_empty()
            || collected_seq.is_empty()
        {
            bail!("one or more templates produced empty feature sequences");
        }

        info!(
            sample_rate = target_sr,
            bite_frames = bite_seq.len(),
            success_frames = success_seq.len(),
            fail_frames = fail_seq.len(),
            collected_frames = collected_seq.len(),
            "audio templates loaded"
        );

        Ok(Self {
            ring,
            _capture: capture,
            extractor,
            live_frames_needed: (target_sr as f32 * cfg.live_seconds) as usize,
            poll_ms: cfg.poll_ms,
            hop: cfg.hop,
            fft_size: cfg.fft_size,
            loudness_target_rms: cfg.loudness_target_rms,
            loudness_gain_min: cfg.loudness_gain_min,
            loudness_gain_max: cfg.loudness_gain_max,
            last_eval_ms: 0,
            bite_matcher: TemplateMatcher::new(
                bite_seq,
                cfg.bite_threshold,
                cfg.min_energy,
                cfg.trigger_cooldown_ms,
            ),
            success_matcher: TemplateMatcher::new(
                success_seq,
                cfg.success_threshold,
                cfg.min_energy,
                cfg.trigger_cooldown_ms,
            ),
            fail_matcher: TemplateMatcher::new(
                fail_seq,
                cfg.fail_threshold,
                cfg.min_energy,
                cfg.trigger_cooldown_ms,
            ),
            collected_matcher: TemplateMatcher::new(
                collected_seq,
                cfg.collected_threshold,
                cfg.min_energy,
                cfg.trigger_cooldown_ms,
            ),
        })
    }

    pub fn poll(&mut self, now_ms: u64) -> AudioEvents {
        if self.last_eval_ms > 0 && now_ms.saturating_sub(self.last_eval_ms) < self.poll_ms {
            return AudioEvents {
                bite_hit: false,
                success_hit: false,
                fail_hit: false,
                collected_hit: false,
                bite_similarity: self.bite_matcher.last_similarity,
                success_similarity: self.success_matcher.last_similarity,
                fail_similarity: self.fail_matcher.last_similarity,
                collected_similarity: self.collected_matcher.last_similarity,
            };
        }
        self.last_eval_ms = now_ms;

        let (samples, channels, total_frames_written) = if let Ok(lock) = self.ring.lock() {
            (
                lock.latest(self.live_frames_needed),
                lock.channels(),
                lock.total_frames(),
            )
        } else {
            (Vec::new(), 0, 0)
        };

        if channels == 0 || samples.is_empty() {
            return AudioEvents {
                bite_hit: false,
                success_hit: false,
                fail_hit: false,
                collected_hit: false,
                bite_similarity: self.bite_matcher.last_similarity,
                success_similarity: self.success_matcher.last_similarity,
                fail_similarity: self.fail_matcher.last_similarity,
                collected_similarity: self.collected_matcher.last_similarity,
            };
        }

        let mono_raw = interleaved_to_mono(&samples, channels);
        let energy = frame_energy(&mono_raw);
        let mono = loudness_match(
            &mono_raw,
            self.loudness_target_rms,
            self.loudness_gain_min,
            self.loudness_gain_max,
        );
        let live_seq = self.extractor.sequence_from_samples(&mono);
        let window_start_frame = total_frames_written.saturating_sub(mono.len() as u64);

        let bite_hit =
            self.bite_matcher
                .update(&live_seq, energy, now_ms, window_start_frame, self.hop, self.fft_size);
        let success_hit = self.success_matcher.update(
            &live_seq,
            energy,
            now_ms,
            window_start_frame,
            self.hop,
            self.fft_size,
        );
        let fail_hit =
            self.fail_matcher
                .update(&live_seq, energy, now_ms, window_start_frame, self.hop, self.fft_size);
        let collected_hit =
            self.collected_matcher
                .update(&live_seq, energy, now_ms, window_start_frame, self.hop, self.fft_size);

        AudioEvents {
            bite_hit,
            success_hit,
            fail_hit,
            collected_hit,
            bite_similarity: self.bite_matcher.last_similarity,
            success_similarity: self.success_matcher.last_similarity,
            fail_similarity: self.fail_matcher.last_similarity,
            collected_similarity: self.collected_matcher.last_similarity,
        }
    }
}

#[derive(Clone)]
struct DecodedAudio {
    mono: Vec<f32>,
    sample_rate: u32,
}

fn decode_audio(path: &Path) -> Result<DecodedAudio> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "wav" => decode_wav(path),
        "ogg" => decode_ogg(path),
        _ => bail!("unsupported audio template format: {}", path.display()),
    }
}

fn decode_wav(path: &Path) -> Result<DecodedAudio> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open wav: {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|v| v as f32 / max_val)
                .collect()
        }
    };

    Ok(DecodedAudio {
        mono: downmix_to_mono(&raw, channels),
        sample_rate: spec.sample_rate,
    })
}

fn decode_ogg(path: &Path) -> Result<DecodedAudio> {
    let file =
        File::open(path).with_context(|| format!("failed to open ogg: {}", path.display()))?;
    let mut rdr = OggStreamReader::new(file)
        .with_context(|| format!("failed to parse ogg: {}", path.display()))?;
    let channels = rdr.ident_hdr.audio_channels as usize;
    let sample_rate = rdr.ident_hdr.audio_sample_rate;
    let mut raw = Vec::<f32>::new();

    while let Some(pkt) = rdr.read_dec_packet_itl()? {
        raw.extend(pkt.iter().map(|v| *v as f32 / i16::MAX as f32));
    }

    Ok(DecodedAudio {
        mono: downmix_to_mono(&raw, channels),
        sample_rate,
    })
}

fn downmix_to_mono(raw_interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return raw_interleaved.to_vec();
    }

    let frame_count = raw_interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
        let base = i * channels;
        let mut sum = 0.0;
        for c in 0..channels {
            sum += raw_interleaved[base + c];
        }
        mono.push(sum / channels as f32);
    }
    mono
}

fn interleaved_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }

    let frames = samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let base = i * channels;
        let mut sum = 0.0;
        for c in 0..channels {
            sum += samples[base + c];
        }
        mono.push(sum / channels as f32);
    }
    mono
}

fn frame_energy(mono: &[f32]) -> f32 {
    if mono.is_empty() {
        return 0.0;
    }
    mono.iter().map(|v| v * v).sum::<f32>() / mono.len() as f32
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let i0 = src_pos.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let t = (src_pos - i0 as f64) as f32;
        out.push(input[i0] * (1.0 - t) + input[i1] * t);
    }
    out
}

fn trim_silence(samples: &[f32], cfg: &AudioConfig) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let peak = samples
        .iter()
        .fold(0.0f32, |m, &v| m.max(v.abs()))
        .max(1e-6);
    let th = (peak * cfg.trim_peak_ratio).max(cfg.trim_floor);

    let mut left = 0usize;
    let mut right = samples.len();
    while left < samples.len() && samples[left].abs() < th {
        left += 1;
    }
    while right > left && samples[right - 1].abs() < th {
        right -= 1;
    }

    samples[left..right].to_vec()
}

fn take_prefix_seconds(samples: &[f32], sample_rate: u32, seconds: f32) -> Vec<f32> {
    if samples.is_empty() || sample_rate == 0 || seconds <= 0.0 {
        return samples.to_vec();
    }
    let need = ((sample_rate as f32) * seconds).round() as usize;
    if need == 0 || samples.len() <= need {
        return samples.to_vec();
    }
    samples[..need].to_vec()
}

fn loudness_match_cfg(samples: &[f32], cfg: &AudioConfig) -> Vec<f32> {
    loudness_match(
        samples,
        cfg.loudness_target_rms,
        cfg.loudness_gain_min,
        cfg.loudness_gain_max,
    )
}

fn loudness_match(samples: &[f32], target_rms: f32, gain_min: f32, gain_max: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
    if rms <= 1e-7 {
        return samples.to_vec();
    }

    // Keep gain bounded so we don't amplify noise/silence too much.
    let gain = (target_rms / rms).clamp(gain_min, gain_max);
    samples
        .iter()
        .map(|v| (v * gain).clamp(-1.0, 1.0))
        .collect()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn subseq_dtw_distance(template: &[Vec<f32>], live: &[Vec<f32>]) -> Option<(f32, usize)> {
    let m = template.len();
    let n = live.len();
    if m == 0 || n == 0 {
        return None;
    }

    let inf = f32::INFINITY;
    let mut prev = vec![0.0f32; n + 1];
    let mut curr = vec![inf; n + 1];

    for i in 1..=m {
        curr[0] = inf;
        for j in 1..=n {
            let cost = l2(&template[i - 1], &live[j - 1]);
            let best_prev = prev[j - 1].min(prev[j]).min(curr[j - 1]);
            curr[j] = cost + best_prev;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let best = prev[1..].iter().fold(inf, |acc, &v| acc.min(v));
    if !best.is_finite() {
        return None;
    }

    let mut best_j = 1usize;
    let mut best_v = prev[1];
    for (idx, &v) in prev[1..].iter().enumerate() {
        if v < best_v {
            best_v = v;
            best_j = idx + 1;
        }
    }

    Some((best / m as f32, best_j - 1))
}



