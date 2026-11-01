//! Tiling and normalisation. Must match train/common.py exactly (integer layout math, no resize).
use image::{Rgb, RgbImage};

pub const TILE: u32 = 224;
pub const MAX_TILES: usize = 8;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn axis_positions(size: u32) -> Vec<u32> {
    if size <= TILE {
        return vec![0];
    }
    let n = size.div_ceil(TILE);
    (0..n).map(|i| (i * (size - TILE) + (n - 1) / 2) / (n - 1)).collect()
}

/// Up to MAX_TILES native-resolution 224px tiles spread over the page, as NCHW f32 (flattened).
/// Returns (data, tile_count).
pub fn page_tiles(img: &RgbImage) -> (Vec<f32>, usize) {
    let padded;
    let img = if img.width() < TILE || img.height() < TILE {
        let mut canvas = RgbImage::from_pixel(img.width().max(TILE), img.height().max(TILE), Rgb([255, 255, 255]));
        image::imageops::replace(&mut canvas, img, 0, 0);
        padded = canvas;
        &padded
    } else {
        img
    };
    let (w, h) = img.dimensions();
    let mut boxes: Vec<(u32, u32)> = Vec::new();
    for y in axis_positions(h) {
        for x in axis_positions(w) {
            boxes.push((x, y));
        }
    }
    if boxes.len() > MAX_TILES {
        let len = boxes.len();
        boxes = (0..MAX_TILES).map(|k| boxes[(k * len) / MAX_TILES]).collect();
    }
    let t = TILE as usize;
    let mut data = vec![0f32; boxes.len() * 3 * t * t];
    for (n, (x0, y0)) in boxes.iter().enumerate() {
        for y in 0..t {
            for x in 0..t {
                let p = img.get_pixel(x0 + x as u32, y0 + y as u32).0;
                for c in 0..3 {
                    data[((n * 3 + c) * t + y) * t + x] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
                }
            }
        }
    }
    (data, boxes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_cover_axis() {
        assert_eq!(axis_positions(100), vec![0]);
        assert_eq!(axis_positions(224), vec![0]);
        assert_eq!(axis_positions(448), vec![0, 224]);
        let p = axis_positions(500);
        assert_eq!(*p.first().unwrap(), 0);
        assert_eq!(*p.last().unwrap(), 500 - TILE);
    }

    #[test]
    fn tile_count_is_capped_and_small_images_are_padded() {
        let big = RgbImage::from_pixel(2000, 3000, Rgb([10, 20, 30]));
        let (data, n) = page_tiles(&big);
        assert_eq!(n, MAX_TILES);
        assert_eq!(data.len(), n * 3 * 224 * 224);
        let small = RgbImage::from_pixel(50, 60, Rgb([0, 0, 0]));
        assert_eq!(page_tiles(&small).1, 1);
    }
}
