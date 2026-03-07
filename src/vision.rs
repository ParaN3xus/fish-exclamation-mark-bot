use anyhow::{Result, anyhow};
use opencv::core::{self, CV_8UC4, Mat, Point, Rect, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;
use ort::ep::CUDA;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::PathBuf;
use tracing::{info, warn};

use crate::types::{BBox, Kp, OuterDet};

pub struct YoloOrt {
    session: Session,
    imgsz: i32,
    class_id: i32,
}

pub fn init_ort_runtime() -> Result<()> {
    let raw = std::env::var("ORT_DYLIB_PATH")
        .ok()
        .unwrap_or("onnxruntime.dll".to_owned());
    let raw_path = PathBuf::from(raw);
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let p = if raw_path.is_absolute() {
        raw_path
    } else {
        exe_dir.join(raw_path)
    };
    if p.is_file() {
        let abs = p.canonicalize().unwrap_or(p);
        info!(path = %abs.display(), "ORT initialized from ORT_DYLIB_PATH");
        let _ = ort::init_from(abs.to_string_lossy().to_string())
            .map_err(|e| anyhow!("{e}"))?
            .commit();
        return Ok(());
    }

    Err(anyhow!(
        "onnxruntime.dll not found; set ORT_DYLIB_PATH to an absolute path or exe-dir-relative path"
    ))
}

impl YoloOrt {
    pub fn new(model_path: &str, imgsz: i32, class_id: i32) -> Result<Self> {
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow!("model not found: {}", model_path));
        }

        let session = match Session::builder()
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .with_execution_providers([CUDA::default().build()])
        {
            Ok(mut b) => {
                info!("trying CUDAExecutionProvider");
                b.commit_from_file(model_path).map_err(|e| anyhow!("{e}"))?
            }
            Err(e) => {
                warn!(error = %e, "CUDA EP unavailable, fallback to CPU");
                Session::builder()
                    .map_err(|e| anyhow!("{e}"))?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(|e| anyhow!("{e}"))?
                    .commit_from_file(model_path)
                    .map_err(|e| anyhow!("{e}"))?
            }
        };

        Ok(Self {
            session,
            imgsz,
            class_id,
        })
    }

    fn centered_square(frame: &Mat) -> Result<(Rect, Mat)> {
        let w = frame.cols();
        let h = frame.rows();
        if w <= 0 || h <= 0 {
            return Err(anyhow!("invalid frame size: {}x{}", w, h));
        }
        let side = w.min(h);
        if side <= 0 {
            return Err(anyhow!("invalid square side from frame size: {}x{}", w, h));
        }
        let sx = (w - side) / 2;
        let sy = (h - side) / 2;
        let r = Rect::new(sx, sy, side, side);
        let sq = Mat::roi(frame, r)?;
        Ok((r, sq.try_clone()?))
    }

    fn preprocess(&self, sq_bgr: &Mat) -> Result<Vec<f32>> {
        let mut resized = Mat::default();
        imgproc::resize(
            sq_bgr,
            &mut resized,
            Size::new(self.imgsz, self.imgsz),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut rgb = Mat::default();
        imgproc::cvt_color(
            &resized,
            &mut rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let data = rgb.data_bytes()?;
        let s = self.imgsz as usize;
        let mut input = vec![0f32; 3 * s * s];
        for y in 0..s {
            let row_off = y * s * 3;
            for x in 0..s {
                let i = row_off + x * 3;
                let chw0 = y * s + x;
                let chw1 = s * s + chw0;
                let chw2 = 2 * s * s + chw0;
                input[chw0] = data[i] as f32 / 255.0;
                input[chw1] = data[i + 1] as f32 / 255.0;
                input[chw2] = data[i + 2] as f32 / 255.0;
            }
        }
        Ok(input)
    }

    pub fn detect_outer(&mut self, frame_bgr: &Mat, conf_thres: f32) -> Result<Option<OuterDet>> {
        let (sq_rect, sq) = Self::centered_square(frame_bgr)?;
        let input = self.preprocess(&sq)?;
        let input_tensor = Tensor::<f32>::from_array((
            vec![1i64, 3, self.imgsz as i64, self.imgsz as i64],
            input,
        ))?;
        let inputs = ort::inputs![input_tensor];
        let outputs = self.session.run(inputs)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[2] < 12 {
            return Ok(None);
        }

        let n = shape[1] as usize;
        let d = shape[2] as usize;
        let side = sq_rect.width as f32;
        let scale = side / self.imgsz as f32;
        let mut best: Option<OuterDet> = None;

        for i in 0..n {
            let base = i * d;
            let conf = data[base + 4];
            let cls = data[base + 5] as i32;
            if cls != self.class_id || conf < conf_thres {
                continue;
            }

            let x1 = data[base] * scale + sq_rect.x as f32;
            let y1 = data[base + 1] * scale + sq_rect.y as f32;
            let x2 = data[base + 2] * scale + sq_rect.x as f32;
            let y2 = data[base + 3] * scale + sq_rect.y as f32;
            let k0x = data[base + 6] * scale + sq_rect.x as f32;
            let k0y = data[base + 7] * scale + sq_rect.y as f32;
            let k1x = data[base + 9] * scale + sq_rect.x as f32;
            let k1y = data[base + 10] * scale + sq_rect.y as f32;

            let b = BBox {
                x: x1.floor() as i32,
                y: y1.floor() as i32,
                w: (x2 - x1).max(1.0).round() as i32,
                h: (y2 - y1).max(1.0).round() as i32,
            };
            let det = OuterDet {
                b,
                top: Kp { x: k0x, y: k0y },
                bot: Kp { x: k1x, y: k1y },
                conf,
            };
            if best.as_ref().map(|v| v.conf).unwrap_or(-1.0) < conf {
                best = Some(det);
            }
        }

        Ok(best)
    }
}

pub fn clip_box(mut b: BBox, w: i32, h: i32) -> BBox {
    b.x = b.x.clamp(0, w - 1);
    b.y = b.y.clamp(0, h - 1);
    b.w = b.w.clamp(1, w - b.x);
    b.h = b.h.clamp(1, h - b.y);
    b
}

pub fn roi_from_outer_and_kp(outer: BBox, top: Kp, bot: Kp, w: i32, h: i32, pad: i32) -> BBox {
    let mut x0 = outer.x.min(top.x.floor() as i32).min(bot.x.floor() as i32) - pad;
    let mut y0 = outer.y.min(top.y.floor() as i32).min(bot.y.floor() as i32) - pad;
    let mut x1 = (outer.x + outer.w)
        .max(top.x.ceil() as i32)
        .max(bot.x.ceil() as i32)
        + pad;
    let mut y1 = (outer.y + outer.h)
        .max(top.y.ceil() as i32)
        .max(bot.y.ceil() as i32)
        + pad;
    x0 = x0.clamp(0, w - 1);
    y0 = y0.clamp(0, h - 1);
    x1 = x1.clamp(x0 + 1, w);
    y1 = y1.clamp(y0 + 1, h);
    BBox {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}

fn build_strip_by_keypoints(
    gray: &Mat,
    outer: BBox,
    top: Kp,
    bot: Kp,
    strip_w: i32,
) -> Result<Option<(Mat, [f64; 6], i32)>> {
    let (mut x0, mut y0, mut x1, mut y1) = (top.x, top.y, bot.x, bot.y);
    if y0 > y1 {
        (x0, y0, x1, y1) = (x1, y1, x0, y0);
    }

    let vx = x1 - x0;
    let vy = y1 - y0;
    if vx * vx + vy * vy < 64.0 {
        return Ok(None);
    }

    let rot_deg = (-vx).atan2(vy) * 180.0 / std::f32::consts::PI;
    let h = gray.rows();
    let w = gray.cols();
    let cx = (x0 + x1) * 0.5;
    let cy = (y0 + y1) * 0.5;

    let mut m = imgproc::get_rotation_matrix_2d(core::Point2f::new(cx, cy), rot_deg as f64, 1.0)?;
    let cosv = (*m.at_2d::<f64>(0, 0)?).abs();
    let sinv = (*m.at_2d::<f64>(0, 1)?).abs();
    let new_w = ((h as f64) * sinv + (w as f64) * cosv).round() as i32;
    let new_h = ((h as f64) * cosv + (w as f64) * sinv).round() as i32;
    *m.at_2d_mut::<f64>(0, 2)? += new_w as f64 * 0.5 - cx as f64;
    *m.at_2d_mut::<f64>(1, 2)? += new_h as f64 * 0.5 - cy as f64;

    let mut rotated = Mat::default();
    imgproc::warp_affine(
        gray,
        &mut rotated,
        &m,
        Size::new(new_w, new_h),
        imgproc::INTER_LINEAR,
        core::BORDER_REPLICATE,
        Scalar::default(),
    )?;

    let m00 = *m.at_2d::<f64>(0, 0)?;
    let m01 = *m.at_2d::<f64>(0, 1)?;
    let m02 = *m.at_2d::<f64>(0, 2)?;
    let m10 = *m.at_2d::<f64>(1, 0)?;
    let m11 = *m.at_2d::<f64>(1, 1)?;
    let m12 = *m.at_2d::<f64>(1, 2)?;
    let tf = |x: f32, y: f32| -> (f64, f64) {
        (
            m00 * x as f64 + m01 * y as f64 + m02,
            m10 * x as f64 + m11 * y as f64 + m12,
        )
    };

    let r0 = tf(x0, y0);
    let r1 = tf(x1, y1);
    let x_center = ((r0.0 + r1.0) * 0.5).round() as i32;

    let corners = [
        (outer.x as f32, outer.y as f32),
        ((outer.x + outer.w) as f32, outer.y as f32),
        ((outer.x + outer.w) as f32, (outer.y + outer.h) as f32),
        (outer.x as f32, (outer.y + outer.h) as f32),
    ];
    let mut miny = f64::INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    for (x, y) in corners {
        let (_, ry) = tf(x, y);
        miny = miny.min(ry);
        maxy = maxy.max(ry);
    }

    let mut y_top = miny.floor() as i32 - 2;
    let mut y_bot = maxy.ceil() as i32 + 2;
    y_top = y_top.clamp(0, new_h - 1);
    y_bot = y_bot.clamp(y_top + 1, new_h);
    let h_strip = y_bot - y_top;
    if h_strip < 8 {
        return Ok(None);
    }

    let x_left = x_center - strip_w / 2;
    let x_right = x_left + strip_w;
    let pad_l = (-x_left).max(0);
    let pad_r = (x_right - new_w).max(0);
    let x_left_c = x_left.max(0);
    let x_right_c = x_right.min(new_w);

    let mut strip = Mat::roi(
        &rotated,
        Rect::new(x_left_c, y_top, (x_right_c - x_left_c).max(1), h_strip),
    )?
    .try_clone()?;

    if pad_l > 0 || pad_r > 0 {
        let mut tmp = Mat::default();
        core::copy_make_border(
            &strip,
            &mut tmp,
            0,
            0,
            pad_l,
            pad_r,
            core::BORDER_REPLICATE,
            Scalar::default(),
        )?;
        strip = tmp;
    }

    if strip.cols() != strip_w {
        let mut tmp = Mat::default();
        imgproc::resize(
            &strip,
            &mut tmp,
            Size::new(strip_w, strip.rows()),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        strip = tmp;
    }

    let mut inv_m = Mat::default();
    imgproc::invert_affine_transform(&m, &mut inv_m)?;
    let a00 = *inv_m.at_2d::<f64>(0, 0)?;
    let a01 = *inv_m.at_2d::<f64>(0, 1)?;
    let a02 = *inv_m.at_2d::<f64>(0, 2)?;
    let a10 = *inv_m.at_2d::<f64>(1, 0)?;
    let a11 = *inv_m.at_2d::<f64>(1, 1)?;
    let a12 = *inv_m.at_2d::<f64>(1, 2)?;

    let s2i = [
        a00,
        a01,
        a00 * x_left as f64 + a01 * y_top as f64 + a02,
        a10,
        a11,
        a10 * x_left as f64 + a11 * y_top as f64 + a12,
    ];

    Ok(Some((strip, s2i, h_strip)))
}

fn map_norm_box_via_strip(norm_box: BBox, s2i: &[f64; 6], strip_h: i32) -> BBox {
    let sy = strip_h as f64 / 168.0;
    let x0 = norm_box.x as f64;
    let y0 = norm_box.y as f64 * sy;
    let x1 = (norm_box.x + norm_box.w) as f64;
    let y1 = (norm_box.y + norm_box.h) as f64 * sy;

    let pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let mut minx = f64::INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    for (x, y) in pts {
        let rx = s2i[0] * x + s2i[1] * y + s2i[2];
        let ry = s2i[3] * x + s2i[4] * y + s2i[5];
        minx = minx.min(rx);
        miny = miny.min(ry);
        maxx = maxx.max(rx);
        maxy = maxy.max(ry);
    }

    BBox {
        x: minx.floor() as i32,
        y: miny.floor() as i32,
        w: (maxx.ceil() - minx.floor()).max(1.0) as i32,
        h: (maxy.ceil() - miny.floor()).max(1.0) as i32,
    }
}

fn patch_spec(norm: &Mat, x: i32, y: i32, k: i32) -> Result<Vec<f32>> {
    let mut v = vec![0.0f32; (k * k) as usize];
    let mut sum = 0.0f32;
    for yy in 0..k {
        for xx in 0..k {
            let p = *norm.at_2d::<u8>(y + yy, x + xx)? as f32;
            let i = (yy * k + xx) as usize;
            v[i] = p;
            sum += p;
        }
    }

    let mean = sum / (k * k) as f32;
    for x in &mut v {
        *x -= mean;
    }

    let mut n = 0.0f32;
    for x in &v {
        n += *x * *x;
    }
    n = n.sqrt().max(1e-6);

    for x in &mut v {
        *x /= n;
    }

    Ok(v)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

pub fn detect_bright_fish_strategy(
    roi_gray: &Mat,
    outer_local: BBox,
    top_local: Kp,
    bot_local: Kp,
    bright_h_hint: Option<i32>,
    fish_spec_hint: Option<&[f32]>,
    proc_hint: Option<(i32, i32)>,
) -> Result<(Option<BBox>, Option<BBox>, i32, i32, Option<Vec<f32>>)> {
    let Some((strip, s2i, strip_h)) =
        build_strip_by_keypoints(roi_gray, outer_local, top_local, bot_local, 8)?
    else {
        return Ok((None, None, 0, 167, None));
    };

    let mut norm = Mat::default();
    imgproc::resize(
        &strip,
        &mut norm,
        Size::new(8, 168),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let proc_top = proc_hint.map(|p| p.0).unwrap_or(0).clamp(0, 167);
    let proc_bot = proc_hint.map(|p| p.1).unwrap_or(167).clamp(proc_top, 167);

    let mut fish_norm: Option<BBox> = None;
    if let Some(spec) = fish_spec_hint {
        let k = 6;
        let mut best = -1e9f32;
        let mut best_xy = (0, proc_top);
        for y in proc_top..=(proc_bot - k + 1).max(proc_top) {
            for x in 0..=(8 - k) {
                let f = patch_spec(&norm, x, y, k)?;
                let s = dot(spec, &f);
                if s > best {
                    best = s;
                    best_xy = (x, y);
                }
            }
        }
        fish_norm = Some(BBox {
            x: best_xy.0,
            y: best_xy.1,
            w: k,
            h: k,
        });
    }

    let mut blur = Mat::default();
    imgproc::gaussian_blur(
        &norm,
        &mut blur,
        Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut bw = Mat::default();
    imgproc::threshold(
        &blur,
        &mut bw,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;

    let mut bw2 = Mat::default();
    let k =
        imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(1, 5), Point::new(-1, -1))?;
    imgproc::morphology_ex(
        &bw,
        &mut bw2,
        imgproc::MORPH_CLOSE,
        &k,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let mut active = vec![false; 168];
    for y in 0..168 {
        if y < proc_top || y > proc_bot {
            continue;
        }
        let mut s = 0;
        for x in 0..8 {
            if *bw2.at_2d::<u8>(y, x)? > 0 {
                s += 1;
            }
        }
        active[y as usize] = s >= 2;
    }

    let fish_cy = fish_norm.map(|f| f.y + f.h / 2);
    let mut cands: Vec<(i32, i32)> = Vec::new();
    for y in (proc_top + 1)..=proc_bot {
        let a0 = active[(y - 1) as usize];
        let a1 = active[y as usize];
        if !a0 && a1 {
            if fish_cy.map(|cy| (y - cy).abs() > 10).unwrap_or(true) {
                cands.push((y, 1));
            }
        }
        if a0 && !a1 {
            let yy = y - 1;
            if fish_cy.map(|cy| (yy - cy).abs() > 10).unwrap_or(true) {
                cands.push((yy, -1));
            }
        }
    }

    let mut bright_norm: Option<BBox> = None;
    if let Some(hh) = bright_h_hint {
        let mut chosen: Option<(i32, i32)> = None;
        if !cands.is_empty() {
            let cand_score = |y: i32, pol: i32| -> i32 {
                let mut len = 0;
                if pol > 0 {
                    let mut yy = y;
                    while yy <= proc_bot && active[yy as usize] {
                        len += 1;
                        yy += 1;
                    }
                } else {
                    let mut yy = y;
                    while yy >= proc_top && active[yy as usize] {
                        len += 1;
                        yy -= 1;
                    }
                }
                len
            };

            let (y, pol) = cands
                .iter()
                .copied()
                .max_by_key(|(yy, pp)| cand_score(*yy, *pp))
                .unwrap_or(cands[0]);
            if pol > 0 {
                let t = y.clamp(proc_top, proc_bot);
                let b = (t + hh - 1).clamp(proc_top, proc_bot);
                chosen = Some((t, b));
            } else {
                let b = y.clamp(proc_top, proc_bot);
                let t = (b - hh + 1).clamp(proc_top, proc_bot);
                chosen = Some((t, b));
            }
        }

        if let Some((t, b)) = chosen {
            if b > t {
                bright_norm = Some(BBox {
                    x: 0,
                    y: t,
                    w: 8,
                    h: b - t + 1,
                });
            }
        }
    } else {
        let mut runs: Vec<(i32, i32)> = Vec::new();
        let mut in_run = false;
        let mut s0 = proc_top;
        for y in proc_top..=proc_bot {
            if active[y as usize] && !in_run {
                in_run = true;
                s0 = y;
            }
            if !active[y as usize] && in_run {
                in_run = false;
                runs.push((s0, y - 1));
            }
        }
        if in_run {
            runs.push((s0, proc_bot));
        }

        let mut merged: Vec<(i32, i32)> = Vec::new();
        for r in runs {
            if let Some(last) = merged.last_mut() {
                if r.0 - last.1 - 1 <= 20 {
                    last.1 = r.1;
                } else {
                    merged.push(r);
                }
            } else {
                merged.push(r);
            }
        }

        if let Some((t, b)) = merged.into_iter().max_by_key(|r| r.1 - r.0) {
            if b > t {
                bright_norm = Some(BBox {
                    x: 0,
                    y: t,
                    w: 8,
                    h: b - t + 1,
                });
            }
        }
    }

    let mut learned_fish_spec = None;
    if fish_spec_hint.is_none() {
        if let Some(br) = bright_norm {
            let cy = (br.y + br.h / 2).clamp(0, 167);
            let x0 = 1;
            let y0 = (cy - 3).clamp(0, 162);
            learned_fish_spec = Some(patch_spec(&norm, x0, y0, 6)?);
            fish_norm = Some(BBox {
                x: x0,
                y: y0,
                w: 6,
                h: 6,
            });
        }
    }

    let br_abs = bright_norm.map(|b| map_norm_box_via_strip(b, &s2i, strip_h));
    let fish_abs = fish_norm.map(|b| map_norm_box_via_strip(b, &s2i, strip_h));
    Ok((br_abs, fish_abs, proc_top, proc_bot, learned_fish_spec))
}

pub fn mat_bgra_from_bytes(w: i32, h: i32, bgra: &[u8]) -> Result<Mat> {
    if w <= 0 || h <= 0 {
        return Err(anyhow!("invalid bgra frame size: {}x{}", w, h));
    }
    let expected = (w as usize)
        .saturating_mul(h as usize)
        .saturating_mul(4);
    if bgra.len() != expected {
        return Err(anyhow!(
            "bgra length mismatch: got {}, expected {} for {}x{}",
            bgra.len(),
            expected,
            w,
            h
        ));
    }
    let mut m = Mat::new_rows_cols_with_default(h, w, CV_8UC4, Scalar::default())?;
    m.data_bytes_mut()?.copy_from_slice(bgra);
    Ok(m)
}
