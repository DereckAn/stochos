//! Pure computer-vision detector (always-available fallback; `image` +
//! `imageproc` only). Pipeline: resize to the logical screen resolution →
//! grayscale → Canny → wider-than-tall dilation (joins a word's letters, keeps
//! lines apart) → connected components → size filtering in logical pixels →
//! drop boxes nested inside a control. Sees visual structure, not semantic
//! clickability.

use anyhow::{bail, Context};
use image::{GrayImage, Luma};
use imageproc::region_labelling::{connected_components, Connectivity};
use std::collections::HashMap;

use super::detect::{DetectorOutput, HintCandidate, HintDetector};
use crate::{backend::Backend, config::config};

#[derive(Default)]
pub struct CvHintDetector;

impl HintDetector for CvHintDetector {
    fn name(&self) -> &'static str {
        "cv"
    }

    fn detect(&self, backend: &mut dyn Backend) -> anyhow::Result<DetectorOutput> {
        let capture = backend.capture_screen().context(
            "hint mode requires screen capture; the CV detector cannot run on this backend",
        )?;
        let (logical_w, _) = backend.screen_size();
        let candidates = detect_candidates_cv(&capture.bgra, capture.w, capture.h, logical_w)?;
        Ok(DetectorOutput {
            candidates,
            capture_w: capture.w,
            capture_h: capture.h,
            focus_rect: None,
        })
    }
}

/// A component in the downscaled working image: its box plus how many mask
/// pixels it actually occupies (used for the solidity score).
struct Component {
    bbox: (u32, u32, u32, u32),
    pixels: u32,
}

/// Run the detector chosen by `hint.improved_detection`.
fn detect_candidates_cv(
    bgra: &[u8],
    capture_w: u32,
    capture_h: u32,
    logical_w: u32,
) -> anyhow::Result<Vec<HintCandidate>> {
    if config().hint.improved_detection {
        detect_improved(bgra, capture_w, capture_h, logical_w)
    } else {
        detect_classic(bgra, capture_w, capture_h)
    }
}

/// The improved detector. `logical_w` is the screen width in logical (scaled)
/// pixels, so size limits mean the same thing on a 1080p screen and a HiDPI
/// one; 0 treats capture pixels as logical.
fn detect_improved(
    bgra: &[u8],
    capture_w: u32,
    capture_h: u32,
    logical_w: u32,
) -> anyhow::Result<Vec<HintCandidate>> {
    if capture_w == 0 || capture_h == 0 {
        bail!("cannot run hint CV detector on an empty capture");
    }
    if bgra.len() < (capture_w as usize * capture_h as usize * 4) {
        bail!("screen capture buffer is smaller than declared dimensions");
    }

    let cfg = &config().hint;
    // Work at the logical resolution: HiDPI captures carry no extra UI, only
    // sharper pixels, and the smaller image is much faster.
    let scale = if logical_w > 0 && logical_w < capture_w {
        logical_w as f32 / capture_w as f32
    } else {
        1.0
    };
    let ww = ((capture_w as f32 * scale).round() as u32).max(1);
    let wh = ((capture_h as f32 * scale).round() as u32).max(1);
    // Working pixels per logical pixel (1 when the capture is HiDPI or 1x).
    let k = if logical_w > capture_w {
        capture_w as f32 / logical_w as f32
    } else {
        1.0
    };

    let gray = grayscale_resized(bgra, capture_w as usize, capture_h as usize, ww, wh);
    let edges = fast_canny(
        &gray,
        cfg.edge_low_threshold as f32,
        cfg.edge_high_threshold as f32,
    );
    // Wider than tall: the letters of a word (and an icon next to its label)
    // join into one target, while lines of text stay separate.
    let joined = dilate_rect(
        &edges,
        ((3.5 * k).round() as u32).max(1),
        ((2.5 * k).round() as u32).max(1),
    );

    let components = connected_component_boxes(&joined);

    // Size limits in logical px. Below: specks, underlines and separators.
    // Above: panels, paragraphs and bars, which are containers, not targets.
    const MIN_W: f32 = 7.0;
    const MIN_H: f32 = 3.0;
    const MAX_W: f32 = 500.0;
    const MAX_H: f32 = 50.0;

    let kept: Vec<(Component, f32)> = components
        .into_iter()
        .filter_map(|component| {
            let (_, _, w, h) = component.bbox;
            let (lw, lh) = (w as f32 / k, h as f32 / k);
            if lw <= MIN_W || lh <= MIN_H || lw >= MAX_W || lh >= MAX_H {
                return None;
            }
            // Coherent controls fill their box; gradient/text speckle scores low.
            let area = (w as f32) * (h as f32);
            let solidity = (component.pixels as f32 / area).clamp(0.0, 1.0);
            Some((component, solidity))
        })
        .collect();

    let kept = drop_icon_inner_boxes(kept, k);

    let inv_scale = 1.0 / scale;
    Ok(kept
        .into_iter()
        .map(|(component, solidity)| {
            let bbox = scale_box_to_capture(component.bbox, inv_scale, capture_w, capture_h);
            HintCandidate {
                bbox,
                score: solidity,
            }
        })
        .collect())
}

/// Canny edge detection, same steps and thresholds as `imageproc::edges::canny`
/// (Gaussian blur with sigma 1.4, 3x3 Sobel, non-maximum suppression,
/// hysteresis) but in integers with direct indexing and no per-pixel `atan2`:
/// about 3x faster, which is what lets the improved detector run at the logical
/// screen resolution. Magnitudes are compared squared, which keeps their order.
fn fast_canny(image: &GrayImage, low_threshold: f32, high_threshold: f32) -> GrayImage {
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w < 3 || h < 3 {
        return GrayImage::new(w as u32, h as u32);
    }
    let blurred = gaussian_blur_7(image.as_raw(), w, h);
    canny_after_blur(&blurred, w, h, low_threshold, high_threshold)
}

/// Sobel, non-maximum suppression and hysteresis on an already blurred image.
fn canny_after_blur(
    blurred: &[u8],
    w: usize,
    h: usize,
    low_threshold: f32,
    high_threshold: f32,
) -> GrayImage {
    let mut out = GrayImage::new(w as u32, h as u32);

    // 3x3 Sobel, magnitude squared. Border pixels never reach the output
    // (suppression and hysteresis skip them), so only the interior is computed.
    let mut gx = vec![0i32; w * h];
    let mut gy = vec![0i32; w * h];
    let mut mag = vec![0u32; w * h];
    for y in 1..h - 1 {
        let (up, row, down) = (
            &blurred[(y - 1) * w..y * w],
            &blurred[y * w..(y + 1) * w],
            &blurred[(y + 1) * w..(y + 2) * w],
        );
        for x in 1..w - 1 {
            let p = |r: &[u8], dx: usize| i32::from(r[x + dx - 1]);
            let sx =
                (p(up, 2) + 2 * p(row, 2) + p(down, 2)) - (p(up, 0) + 2 * p(row, 0) + p(down, 0));
            let sy =
                (p(down, 0) + 2 * p(down, 1) + p(down, 2)) - (p(up, 0) + 2 * p(up, 1) + p(up, 2));
            let i = y * w + x;
            gx[i] = sx;
            gy[i] = sy;
            mag[i] = (sx * sx + sy * sy) as u32;
        }
    }

    // Non-maximum suppression along the gradient, quantized to 0/45/90/135
    // degrees with tan(22.5) = 0.4142 and tan(67.5) = 2.4142 (x1024).
    let mut thin = vec![0u32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let (ax, ay) = (i64::from(gx[i].abs()), i64::from(gy[i].abs()));
            let (n1, n2) = if ay * 1024 < ax * 424 {
                (i - 1, i + 1)
            } else if ay * 1024 > ax * 2472 {
                (i - w, i + w)
            } else if (gx[i] >= 0) == (gy[i] >= 0) {
                (i + w + 1, i - w - 1)
            } else {
                (i + w - 1, i - w + 1)
            };
            if mag[i] >= mag[n1] && mag[i] >= mag[n2] {
                thin[i] = mag[i];
            }
        }
    }

    // Hysteresis: strong edges, plus weak edges connected to them. Walks the
    // same 6 neighbours as imageproc (it never steps up or up-right), so the
    // edge maps, and the targets tuned on them, stay the same; following all
    // 8 keeps more faint edges, which then merge neighbouring targets.
    let low = (low_threshold * low_threshold) as u32;
    let high = (high_threshold * high_threshold) as u32;
    let edges = out.as_mut();
    let mut stack = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if thin[i] < high || edges[i] != 0 {
                continue;
            }
            edges[i] = 255;
            stack.push(i);
            while let Some(j) = stack.pop() {
                let (jx, jy) = (j % w, j / w);
                for (dx, dy) in [(1, 0), (1, 1), (0, 1), (-1, -1), (-1, 0), (-1, 1)] {
                    let (nx, ny) = (jx as isize + dx, jy as isize + dy);
                    if nx < 1 || ny < 1 || nx >= w as isize - 1 || ny >= h as isize - 1 {
                        continue;
                    }
                    let n = ny as usize * w + nx as usize;
                    if edges[n] == 0 && thin[n] >= low {
                        edges[n] = 255;
                        stack.push(n);
                    }
                }
            }
        }
    }
    out
}

/// Separable 7-tap Gaussian (sigma 1.4, as imageproc's kernel), replicated
/// borders, fixed-point weights summing to 4096. Each pass truncates to u8
/// like imageproc does; rounding instead loses faint dark-theme edges.
fn gaussian_blur_7(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    const SIGMA: f32 = 1.4;
    let raw: Vec<f32> = (-3..=3i32)
        .map(|i| (-(i * i) as f32 / (2.0 * SIGMA * SIGMA)).exp())
        .collect();
    let sum: f32 = raw.iter().sum();
    let k: Vec<u32> = raw
        .iter()
        .map(|v| (v / sum * 4096.0).round() as u32)
        .collect();
    let norm: u32 = k.iter().sum();

    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    // Horizontal pass, rounded back to u8 per pixel.
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let acc: u32 = if x >= 3 && x + 3 < w {
                k.iter()
                    .zip(&row[x - 3..x + 4])
                    .map(|(&kv, &p)| kv * u32::from(p))
                    .sum()
            } else {
                k.iter()
                    .enumerate()
                    .map(|(t, &kv)| kv * u32::from(row[clamp(x as isize + t as isize - 3, w)]))
                    .sum()
            };
            tmp[y * w + x] = (acc / norm) as u8;
        }
    }
    // Vertical pass, row by row so memory is read in order.
    let mut acc = vec![0u32; w];
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        acc.iter_mut().for_each(|a| *a = 0);
        for (t, &kv) in k.iter().enumerate() {
            let sy = clamp(y as isize + t as isize - 3, h);
            for (a, &p) in acc.iter_mut().zip(&tmp[sy * w..(sy + 1) * w]) {
                *a += kv * u32::from(p);
            }
        }
        for (o, &a) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = (a / norm) as u8;
        }
    }
    out
}

/// Binary dilation by a `kw` x `kh` rectangle centered on each pixel
/// (separable; a running count per line keeps it O(pixels)).
fn dilate_rect(mask: &GrayImage, kw: u32, kh: u32) -> GrayImage {
    let (w, h) = (mask.width() as usize, mask.height() as usize);
    let rows = dilate_lines(mask.as_raw(), w, h, 1, w, kw as usize);
    let cols = dilate_lines(&rows, h, w, w, 1, kh as usize);
    GrayImage::from_vec(w as u32, h as u32, cols).expect("same dimensions")
}

/// Dilate `count` lines of `len` pixels; pixel `i` of line `l` is at
/// `l * line_step + i * step`.
fn dilate_lines(
    src: &[u8],
    len: usize,
    count: usize,
    step: usize,
    line_step: usize,
    k: usize,
) -> Vec<u8> {
    let (before, after) = ((k.max(1) - 1) / 2, k.max(1) / 2);
    let mut out = vec![0u8; src.len()];
    let mut prefix = vec![0u32; len + 1];
    for l in 0..count {
        let at = |i: usize| l * line_step + i * step;
        for i in 0..len {
            prefix[i + 1] = prefix[i] + u32::from(src[at(i)] != 0);
        }
        for i in 0..len {
            let (lo, hi) = (i.saturating_sub(before), (i + after + 1).min(len));
            if prefix[hi] > prefix[lo] {
                out[at(i)] = 255;
            }
        }
    }
    out
}

/// The original detector (`hint.improved_detection = false`): downscale to
/// `downscale_longest`, Canny, symmetric closing, connected components, and
/// size limits relative to the working image. Kept unchanged so the two can be
/// compared and the improved one stays opt-out.
fn detect_classic(
    bgra: &[u8],
    capture_w: u32,
    capture_h: u32,
) -> anyhow::Result<Vec<HintCandidate>> {
    if capture_w == 0 || capture_h == 0 {
        bail!("cannot run hint CV detector on an empty capture");
    }
    if bgra.len() < (capture_w as usize * capture_h as usize * 4) {
        bail!("screen capture buffer is smaller than declared dimensions");
    }

    let cfg = &config().hint;
    let longest = capture_w.max(capture_h);
    let scale = if cfg.downscale_longest > 0 && longest > cfg.downscale_longest {
        cfg.downscale_longest as f32 / longest as f32
    } else {
        1.0
    };
    let ww = ((capture_w as f32 * scale).round() as u32).max(1);
    let wh = ((capture_h as f32 * scale).round() as u32).max(1);

    let gray = grayscale_resized(bgra, capture_w as usize, capture_h as usize, ww, wh);
    let edges = imageproc::edges::canny(
        &gray,
        cfg.edge_low_threshold as f32,
        cfg.edge_high_threshold as f32,
    );
    // Closing (dilate + erode) reconnects the broken strokes of one element
    // without inflating boxes or bridging neighbours.
    const CLOSE_RADIUS: u8 = 2;
    let closed = imageproc::morphology::close(
        &edges,
        imageproc::distance_transform::Norm::LInf,
        CLOSE_RADIUS,
    );

    let components = connected_component_boxes(&closed);

    // Shape filters, relative to the working image so behaviour matches at
    // 1080p and 4K: minimum side (sub-glyph noise), maximum side/area
    // fractions (bars, panels, banners), maximum elongation (separators,
    // scrollbars, underlines).
    const MIN_SIZE: u32 = 8;
    const MAX_SIZE_FRAC: f32 = 0.6;
    const MAX_AREA_FRAC: f32 = 0.25;
    const MAX_ASPECT: f32 = 12.0;

    let work_area = (ww as f32) * (wh as f32);
    let max_w = (MAX_SIZE_FRAC * ww as f32).max(MIN_SIZE as f32);
    let max_h = (MAX_SIZE_FRAC * wh as f32).max(MIN_SIZE as f32);
    let max_area = (MAX_AREA_FRAC * work_area).max(1.0);

    let kept: Vec<(Component, f32)> = components
        .into_iter()
        .filter_map(|component| {
            let (_, _, w, h) = component.bbox;
            if w < MIN_SIZE || h < MIN_SIZE {
                return None;
            }
            if (w as f32) > max_w || (h as f32) > max_h {
                return None;
            }
            let area = (w as f32) * (h as f32);
            if area > max_area {
                return None;
            }
            let aspect = (w.max(h) as f32) / (w.min(h).max(1) as f32);
            if aspect > MAX_ASPECT {
                return None;
            }
            // Coherent controls fill their box; gradient/text speckle scores low.
            let solidity = (component.pixels as f32 / area).clamp(0.0, 1.0);
            Some((component, solidity))
        })
        .collect();

    let kept = drop_icon_inner_boxes_classic(kept);

    let inv_scale = 1.0 / scale;
    Ok(kept
        .into_iter()
        .map(|(component, solidity)| {
            let bbox = scale_box_to_capture(component.bbox, inv_scale, capture_w, capture_h);
            HintCandidate {
                bbox,
                score: solidity,
            }
        })
        .collect())
}

/// Drop a box sitting concentrically inside a small, near-square box — an
/// icon glyph inside its own hit area, where both labels would point at the
/// same place. Deliberately narrow so container children are preserved.
/// Thresholds are in working-image pixels.
fn drop_icon_inner_boxes_classic(mut items: Vec<(Component, f32)>) -> Vec<(Component, f32)> {
    const MAX_PARENT_SIDE: u32 = 48;
    const SQUARE_TOLERANCE: u32 = 6;
    const CONCENTRIC_TOLERANCE: u32 = 6;

    let boxes: Vec<(u32, u32, u32, u32)> = items.iter().map(|(c, _)| c.bbox).collect();
    let mut drop = vec![false; boxes.len()];

    for (i, &child) in boxes.iter().enumerate() {
        let smallest_parent = boxes
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i && !drop[*j])
            .filter(|(_, parent)| contains(**parent, child))
            .min_by_key(|(_, parent)| parent.2 * parent.3)
            .map(|(_, parent)| *parent);

        let Some(parent) = smallest_parent else {
            continue;
        };
        let near_square = parent.2.abs_diff(parent.3) <= SQUARE_TOLERANCE;
        let small = parent.2 <= MAX_PARENT_SIDE && parent.3 <= MAX_PARENT_SIDE;
        let child_center = (child.0 + child.2 / 2, child.1 + child.3 / 2);
        let parent_center = (parent.0 + parent.2 / 2, parent.1 + parent.3 / 2);
        let concentric = child_center.0.abs_diff(parent_center.0) <= CONCENTRIC_TOLERANCE
            && child_center.1.abs_diff(parent_center.1) <= CONCENTRIC_TOLERANCE;
        if small && near_square && concentric {
            drop[i] = true;
        }
    }

    let mut keep = drop.into_iter();
    items.retain(|_| !keep.next().unwrap_or(false));
    items
}

/// Map a working-image box back to clamped capture-pixel coordinates.
fn scale_box_to_capture(
    (x, y, w, h): (u32, u32, u32, u32),
    inv_scale: f32,
    capture_w: u32,
    capture_h: u32,
) -> (u32, u32, u32, u32) {
    let nx = ((x as f32 * inv_scale).round() as u32).min(capture_w.saturating_sub(1));
    let ny = ((y as f32 * inv_scale).round() as u32).min(capture_h.saturating_sub(1));
    let nw = ((w as f32 * inv_scale).round() as u32)
        .max(1)
        .min(capture_w - nx);
    let nh = ((h as f32 * inv_scale).round() as u32)
        .max(1)
        .min(capture_h - ny);
    (nx, ny, nw, nh)
}

/// Drop a box inside another kept box when both labels would point at the
/// same control: text centered in its button, a glyph inside a small square
/// icon, or the flat strokes of an icon (hamburger lines). `k` is working
/// pixels per logical pixel; thresholds below are logical px.
fn drop_icon_inner_boxes(mut items: Vec<(Component, f32)>, k: f32) -> Vec<(Component, f32)> {
    let px = |logical: f32| (logical * k).round() as u32;
    let max_icon_side = px(40.0);
    let square_tolerance = px(5.0);
    let concentric_tolerance = px(8.0);
    let max_flat_h = px(6.0);

    let boxes: Vec<(u32, u32, u32, u32)> = items.iter().map(|(c, _)| c.bbox).collect();
    let mut drop = vec![false; boxes.len()];

    for (i, &child) in boxes.iter().enumerate() {
        let smallest_parent = boxes
            .iter()
            .enumerate()
            .filter(|(j, parent)| *j != i && **parent != child)
            .filter(|(_, parent)| contains(**parent, child))
            .min_by_key(|(_, parent)| parent.2 * parent.3)
            .map(|(_, parent)| *parent);

        let Some(parent) = smallest_parent else {
            continue;
        };
        let small_square = parent.2.abs_diff(parent.3) <= square_tolerance
            && parent.2 <= max_icon_side
            && parent.3 <= max_icon_side;
        let child_center = (child.0 + child.2 / 2, child.1 + child.3 / 2);
        let parent_center = (parent.0 + parent.2 / 2, parent.1 + parent.3 / 2);
        let concentric = child_center.0.abs_diff(parent_center.0) <= concentric_tolerance
            && child_center.1.abs_diff(parent_center.1) <= concentric_tolerance;
        let flat = child.3 <= max_flat_h;
        if small_square || concentric || flat {
            drop[i] = true;
        }
    }

    let mut keep = drop.into_iter();
    items.retain(|_| !keep.next().unwrap_or(false));
    items
}

fn contains(parent: (u32, u32, u32, u32), child: (u32, u32, u32, u32)) -> bool {
    parent.0 <= child.0
        && parent.1 <= child.1
        && parent.0 + parent.2 >= child.0 + child.2
        && parent.1 + parent.3 >= child.1 + child.3
}

/// Nearest-neighbour downscale of a BGRA capture straight into grayscale —
/// edges survive it, and it is far cheaper than a filtered resize.
fn grayscale_resized(bgra: &[u8], src_w: usize, src_h: usize, dst_w: u32, dst_h: u32) -> GrayImage {
    let dst_w_usize = dst_w as usize;
    let dst_h_usize = dst_h as usize;
    let mut out = vec![0u8; dst_w_usize * dst_h_usize];
    for y in 0..dst_h_usize {
        let src_y = (y * src_h / dst_h_usize).min(src_h - 1);
        for x in 0..dst_w_usize {
            let src_x = (x * src_w / dst_w_usize).min(src_w - 1);
            let off = (src_y * src_w + src_x) * 4;
            let b = bgra[off] as u32;
            let g = bgra[off + 1] as u32;
            let r = bgra[off + 2] as u32;
            out[y * dst_w_usize + x] = ((29 * b + 150 * g + 77 * r) / 256) as u8;
        }
    }
    GrayImage::from_vec(dst_w, dst_h, out).expect("grayscale buffer size matches image dimensions")
}

/// Connected components of a binary mask, with occupied-pixel counts for the
/// solidity score.
fn connected_component_boxes(mask: &GrayImage) -> Vec<Component> {
    let labels = connected_components(mask, Connectivity::Eight, Luma([0u8]));
    // (min_x, min_y, max_x, max_y, pixel_count)
    let mut rects: HashMap<u32, (u32, u32, u32, u32, u32)> = HashMap::new();

    for (x, y, label) in labels.enumerate_pixels() {
        let label = label[0];
        if label == 0 {
            continue;
        }
        rects
            .entry(label)
            .and_modify(|rect| {
                rect.0 = rect.0.min(x);
                rect.1 = rect.1.min(y);
                rect.2 = rect.2.max(x);
                rect.3 = rect.3.max(y);
                rect.4 += 1;
            })
            .or_insert((x, y, x, y, 1));
    }

    rects
        .into_values()
        .map(|(min_x, min_y, max_x, max_y, pixels)| Component {
            bbox: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
            pixels,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solidity_is_pixel_fill_ratio() {
        // A hollow 4x4 ring: 12 of 16 box pixels filled, so solidity = 12/16.
        let mut img = GrayImage::new(4, 4);
        for (x, y, p) in img.enumerate_pixels_mut() {
            if x == 0 || x == 3 || y == 0 || y == 3 {
                *p = Luma([255]);
            }
        }
        let comps = connected_component_boxes(&img);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].pixels, 12);
        assert_eq!(comps[0].bbox, (0, 0, 4, 4));
        let (_, _, w, h) = comps[0].bbox;
        let solidity = comps[0].pixels as f32 / (w * h) as f32;
        assert!((solidity - 12.0 / 16.0).abs() < 1e-6);
    }

    #[test]
    fn fast_canny_matches_imageproc_on_synthetic_ui() {
        // Buttons, faint dark-theme text strokes and a gradient: the cases the
        // integer path must handle like imageproc.
        let (w, h) = (320u32, 200u32);
        let mut img = GrayImage::from_fn(w, h, |x, _| Luma([20 + (x / 8) as u8]));
        for (x0, y0, bw, bh, v) in [
            (20, 20, 80, 30, 200u8),
            (120, 20, 60, 24, 60),
            (40, 90, 200, 2, 90),
        ] {
            for y in y0..y0 + bh {
                for x in x0..x0 + bw {
                    img.put_pixel(x, y, Luma([v]));
                }
            }
        }
        for i in 0..40u32 {
            let (x, y) = (30 + i * 6, 140 + (i * 7) % 20);
            for d in 0..5 {
                img.put_pixel(x + d % 2, y + d, Luma([120]));
            }
        }
        // Compared after the widening the detector applies: the fixed-point
        // blur can move a perfectly sharp synthetic edge by one pixel, which
        // never changes a box.
        let ours = dilate_rect(&fast_canny(&img, 50.0, 150.0), 4, 3);
        let theirs = dilate_rect(&imageproc::edges::canny(&img, 50.0, 150.0), 4, 3);
        let (mut both, mut either) = (0u32, 0u32);
        for (a, b) in ours.as_raw().iter().zip(theirs.as_raw()) {
            both += u32::from(*a != 0 && *b != 0);
            either += u32::from(*a != 0 || *b != 0);
        }
        assert!(either > 100, "test image produced too few edges");
        let agreement = f64::from(both) / f64::from(either);
        assert!(agreement > 0.97, "edge masks diverge: {agreement}");
    }

    #[test]
    fn classic_icon_inner_box_is_dropped_but_container_children_survive() {
        let comp = |bbox: (u32, u32, u32, u32)| {
            (
                Component {
                    bbox,
                    pixels: bbox.2 * bbox.3,
                },
                1.0,
            )
        };
        // Small near-square icon (outer ring) with a concentric inner glyph.
        let icon_outer = comp((100, 100, 24, 24));
        let icon_inner = comp((106, 106, 12, 12));
        // A large container with two real buttons inside it.
        let container = comp((0, 0, 400, 120));
        let button_a = comp((10, 20, 80, 30));
        let button_b = comp((120, 20, 80, 30));

        let kept = drop_icon_inner_boxes_classic(vec![
            icon_outer, icon_inner, container, button_a, button_b,
        ]);
        let boxes: Vec<_> = kept.iter().map(|(c, _)| c.bbox).collect();

        // The concentric inner glyph is gone.
        assert!(!boxes.contains(&(106, 106, 12, 12)));
        // Everything else — including the container's children — is preserved.
        assert!(boxes.contains(&(100, 100, 24, 24)));
        assert!(boxes.contains(&(0, 0, 400, 120)));
        assert!(boxes.contains(&(10, 20, 80, 30)));
        assert!(boxes.contains(&(120, 20, 80, 30)));
    }

    #[test]
    fn dilation_is_wider_than_tall() {
        // One set pixel in the middle of a 9x9 mask, dilated by a 5x3 rectangle.
        let mut img = GrayImage::new(9, 9);
        img.put_pixel(4, 4, Luma([255]));
        let out = dilate_rect(&img, 5, 3);
        let set: Vec<(u32, u32)> = out
            .enumerate_pixels()
            .filter(|(_, _, p)| p[0] != 0)
            .map(|(x, y, _)| (x, y))
            .collect();
        let expected: Vec<(u32, u32)> =
            (3..=5).flat_map(|y| (2..=6).map(move |x| (x, y))).collect();
        assert_eq!(set, expected);
    }

    #[test]
    fn icon_inner_box_is_dropped_but_container_children_survive() {
        let comp = |bbox: (u32, u32, u32, u32)| {
            (
                Component {
                    bbox,
                    pixels: bbox.2 * bbox.3,
                },
                1.0,
            )
        };
        // Small near-square icon (outer ring) with a concentric inner glyph.
        let icon_outer = comp((100, 100, 24, 24));
        let icon_inner = comp((106, 106, 12, 12));
        // A large container with two real buttons inside it.
        let container = comp((0, 0, 400, 120));
        let button_a = comp((10, 20, 80, 30));
        let button_b = comp((120, 20, 80, 30));

        let kept = drop_icon_inner_boxes(
            vec![icon_outer, icon_inner, container, button_a, button_b],
            1.0,
        );
        let boxes: Vec<_> = kept.iter().map(|(c, _)| c.bbox).collect();

        // The concentric inner glyph is gone.
        assert!(!boxes.contains(&(106, 106, 12, 12)));
        // Everything else — including the container's children — is preserved.
        assert!(boxes.contains(&(100, 100, 24, 24)));
        assert!(boxes.contains(&(0, 0, 400, 120)));
        assert!(boxes.contains(&(10, 20, 80, 30)));
        assert!(boxes.contains(&(120, 20, 80, 30)));
    }
}

/// Dev tool, not a test of invariants: run the CV detector over a real
/// screenshot and print every candidate box, to study behaviour on actual UIs:
///   grim -t ppm /tmp/shot.ppm
///   STOCHOS_CV_IMAGE=/tmp/shot.ppm STOCHOS_CV_LOGICAL_W=2560 cargo test --release -- --ignored --nocapture cv_boxes
#[cfg(test)]
mod probe {
    use super::*;

    /// Minimal binary PPM (P6, maxval 255) reader → BGRA buffer.
    fn read_ppm(path: &str) -> (Vec<u8>, u32, u32) {
        let data = std::fs::read(path).expect("read image file");
        let mut fields = Vec::new();
        let mut pos = 0;
        while fields.len() < 4 {
            // Skip whitespace and comments between header fields.
            while pos < data.len() && (data[pos].is_ascii_whitespace() || data[pos] == b'#') {
                if data[pos] == b'#' {
                    while pos < data.len() && data[pos] != b'\n' {
                        pos += 1;
                    }
                }
                pos += 1;
            }
            let start = pos;
            while pos < data.len() && !data[pos].is_ascii_whitespace() {
                pos += 1;
            }
            fields.push(std::str::from_utf8(&data[start..pos]).unwrap().to_owned());
        }
        pos += 1; // single whitespace after maxval
        assert_eq!(fields[0], "P6", "expected binary PPM (grim -t ppm)");
        let (w, h): (u32, u32) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
        let rgb = &data[pos..pos + (w * h * 3) as usize];
        let mut bgra = vec![255u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            bgra[i * 4] = rgb[i * 3 + 2];
            bgra[i * 4 + 1] = rgb[i * 3 + 1];
            bgra[i * 4 + 2] = rgb[i * 3];
        }
        (bgra, w, h)
    }

    #[test]
    #[ignore = "set STOCHOS_CV_IMAGE to a P6 .ppm screenshot"]
    fn fast_canny_matches_imageproc() {
        let path = std::env::var("STOCHOS_CV_IMAGE").expect("set STOCHOS_CV_IMAGE");
        let (bgra, w, h) = read_ppm(&path);
        let gray = grayscale_resized(&bgra, w as usize, h as usize, w / 2, h / 2);
        let t = std::time::Instant::now();
        let ours = fast_canny(&gray, 50.0, 150.0);
        let ours_t = t.elapsed();
        let t = std::time::Instant::now();
        let theirs = imageproc::edges::canny(&gray, 50.0, 150.0);
        let theirs_t = t.elapsed();
        let (mut both, mut either) = (0u64, 0u64);
        for (a, b) in ours.as_raw().iter().zip(theirs.as_raw()) {
            both += u64::from(*a != 0 && *b != 0);
            either += u64::from(*a != 0 || *b != 0);
        }
        let agreement = both as f64 / either.max(1) as f64;
        eprintln!("fast {ours_t:?} vs imageproc {theirs_t:?}, edge agreement {agreement:.4}");
        // The blur rounds in fixed point, so a few faint edge pixels differ.
        assert!(agreement > 0.97, "edge masks diverge: {agreement}");
    }

    #[test]
    #[ignore = "set STOCHOS_CV_IMAGE to a P6 .ppm screenshot"]
    fn cv_boxes_from_screenshot() {
        let path = std::env::var("STOCHOS_CV_IMAGE").expect("set STOCHOS_CV_IMAGE");
        std::env::set_var(
            "XDG_CONFIG_HOME",
            std::env::temp_dir().join("stochos-cv-probe"),
        );
        crate::config::init().expect("load config");
        let (bgra, w, h) = read_ppm(&path);
        let start = std::time::Instant::now();
        let logical_w = std::env::var("STOCHOS_CV_LOGICAL_W")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let candidates = detect_candidates_cv(&bgra, w, h, logical_w).expect("cv detection");
        eprintln!(
            "cv: {} candidates in {:?} on {w}x{h}",
            candidates.len(),
            start.elapsed()
        );
        let mut sorted = candidates;
        sorted.sort_by_key(|c| (c.bbox.1, c.bbox.0));
        for c in &sorted {
            eprintln!("  bbox={:?} score={:.2}", c.bbox, c.score);
        }
    }
}
