use std::sync::mpsc::channel;
use std::thread;

use anyhow::Result;

use image::DynamicImage;
use image::GrayImage;
use image::Luma;
use image::Rgba;
use image::RgbaImage;

use ndarray::concatenate;
use ndarray::s;
use ndarray::Array3;
use ndarray::Axis;

use show_image::WindowProxy;

pub type NDRgbaImage = Array3<f64>;
pub type NDGrayImage = Array3<f64>;

pub struct SplitImage {
    pub r: NDGrayImage,
    pub g: NDGrayImage,
    pub b: NDGrayImage,
    pub a: NDGrayImage,
}

#[allow(dead_code)]
pub enum ImgConversionType {
    CLAMP,
    NORMALIZE,
}

pub fn show_image(img: DynamicImage, name: &str) -> Result<WindowProxy> {
    let window = show_image::create_window(name, Default::default())?;
    window.set_image(name, img)?;
    Ok(window)
}

pub fn show_nd_image(img: &NDRgbaImage, name: &str) -> Result<WindowProxy> {
    let window = show_image::create_window(name, Default::default())?;
    window.set_image(name, ndarray_to_image_rgba(img))?;
    Ok(window)
}

pub fn show_rgb_image(img: RgbaImage, name: &str) -> Result<WindowProxy> {
    show_image(image::DynamicImage::ImageRgba8(img), name)
}

pub fn show_gray_image(img: GrayImage, name: &str) -> Result<WindowProxy> {
    show_image(image::DynamicImage::ImageLuma8(img), name)
}

pub fn wait_for_windows_to_close(windows: Vec<WindowProxy>) -> Result<()> {
    if windows.len() == 0 {
        return Ok(());
    }
    let (tx, rx) = channel();
    for window in &windows {
        let _tx = tx.clone();
        let event_receiver = window.event_channel()?;
        thread::spawn(move || loop {
            if let Ok(show_image::event::WindowEvent::KeyboardInput(event)) = event_receiver.recv()
            {
                if !event.is_synthetic && event.input.state.is_pressed() {
                    println!("Key pressed!");
                    if let Err(err) = _tx.send(()) {
                        println!(
                            "wait_for_windows_to_close: failed to send keypress event to channel: {:?}", err
                        );
                    }
                    break;
                }
            }
        });
    }
    rx.recv()?;
    Ok(())
}

pub fn load_rgba_img_from_file(path: &str) -> Result<RgbaImage> {
    let img = image::open(path)?;
    Ok(img.into_rgba8())
}

pub fn load_gray_img_from_file(path: &str) -> Result<GrayImage> {
    let img = image::open(path)?;
    Ok(img.into_luma8())
}

pub fn load_nd_rgb_img_from_file(path: &str) -> Result<NDRgbaImage> {
    let img = load_rgba_img_from_file(path)?;
    Ok(image_to_ndarray_rgba(&img))
}

pub fn load_nd_gray_img_from_file(path: &str) -> Result<NDGrayImage> {
    let img = load_gray_img_from_file(path)?;
    Ok(image_to_ndarray_gray(&img))
}

pub fn write_nd_rgba_img_to_file(path: &str, img: &NDRgbaImage) -> Result<()> {
    ndarray_to_image_rgba(img).save_with_format(path, image::ImageFormat::Png)?;
    Ok(())
}

pub fn write_nd_gray_img_to_file(
    path: &str,
    img: &NDGrayImage,
    conversion_type: ImgConversionType,
) -> Result<()> {
    ndarray_to_image_gray(img, conversion_type).save_with_format(path, image::ImageFormat::Png)?;
    Ok(())
}

pub fn split_img_channels(img: &NDRgbaImage) -> SplitImage {
    let r = img.clone().slice_move(s![.., .., 0..1]);
    let g = img.clone().slice_move(s![.., .., 1..2]);
    let b = img.clone().slice_move(s![.., .., 2..3]);
    let a = img.clone().slice_move(s![.., .., 3..4]);
    SplitImage { r, g, b, a }
}

pub fn merge_img_channels(img: &SplitImage) -> Result<NDRgbaImage> {
    let img = concatenate(
        Axis(2),
        &[img.r.view(), img.g.view(), img.b.view(), img.a.view()],
    )?;
    Ok(img)
}

pub fn rgba_img_to_grayscale(img: &NDRgbaImage) -> NDGrayImage {
    let img_shape = img.shape();
    let mut out = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            // http://www.songho.ca/dsp/luminance/luminance.html
            out[[x, y, 0]] =
                (0.288 * img[[x, y, 0]] + 0.587 * img[[x, y, 1]] + 0.114 * img[[x, y, 2]])
                    * img[[x, y, 3]];
        }
    }
    out
}

pub fn image_to_ndarray_rgba(img: &RgbaImage) -> NDRgbaImage {
    let mut out = Array3::zeros((img.width() as usize, img.height() as usize, 4));
    for x in 0..img.width() as usize {
        for y in 0..img.height() as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            out[[x, y, 0]] = pixel[0] as f64;
            out[[x, y, 1]] = pixel[1] as f64;
            out[[x, y, 2]] = pixel[2] as f64;
            out[[x, y, 3]] = pixel[3] as f64;
        }
    }
    out
}

pub fn image_to_ndarray_gray(img: &GrayImage) -> NDGrayImage {
    let mut out = Array3::zeros((img.width() as usize, img.height() as usize, 0));
    for x in 0..img.width() as usize {
        for y in 0..img.height() as usize {
            let pixel = img.get_pixel(x as u32, y as u32);
            out[[x, y, 0]] = pixel[0] as f64;
        }
    }
    out
}

pub fn ndarray_to_image_rgba(img: &NDRgbaImage) -> RgbaImage {
    let img_shape = img.shape();
    let mut out = RgbaImage::new(img_shape[0] as u32, img_shape[1] as u32);
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let clamp = |val: f64| val.max(0.).min(255.).round() as u8;
            let r = clamp(img[[x, y, 0]]);
            let g = clamp(img[[x, y, 1]]);
            let b = clamp(img[[x, y, 2]]);
            let a = clamp(img[[x, y, 3]]);
            out.put_pixel(x as u32, y as u32, Rgba::from([r, g, b, a]));
        }
    }
    out
}

pub fn ndarray_to_image_gray(img: &NDGrayImage, conversion_type: ImgConversionType) -> GrayImage {
    let img_shape = img.shape();
    let mut out = GrayImage::new(img_shape[0] as u32, img_shape[1] as u32);
    let mut max_val = 0.;
    if let ImgConversionType::NORMALIZE = conversion_type {
        for x in 0..img_shape[0] {
            for y in 0..img_shape[1] {
                let val = img[[x, y, 0]];
                if max_val < val {
                    max_val = val;
                }
            }
        }
    }
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            let convert = |val: f64| match conversion_type {
                ImgConversionType::NORMALIZE => (255. * val.max(0.) / max_val).round() as u8,
                ImgConversionType::CLAMP => val.max(0.).min(255.).round() as u8,
            };
            let r = convert(img[[x, y, 0]]);
            out.put_pixel(x as u32, y as u32, Luma::from([r]));
        }
    }
    out
}

// formula taken from: https://en.wikipedia.org/wiki/Bilinear_interpolation
// TODO: if you give perfect pixels, this returns [NaN, NaN, NaN]
pub fn sample_nd_img(img: &NDRgbaImage, x: f64, y: f64) -> [f64; 4] {
    let x1 = x.floor();
    let x2 = x.ceil();
    let y1 = y.floor();
    let y2 = y.ceil();
    let do_interpolation_for_channel = |channel: usize| {
        let corners = [
            img[[x1 as usize, y1 as usize, channel]],
            img[[x2 as usize, y1 as usize, channel]],
            img[[x1 as usize, y2 as usize, channel]],
            img[[x2 as usize, y2 as usize, channel]],
        ];
        (1.0 / ((x2 - x1) * (y2 - y1)))
            * (corners[0] * ((x2 - x) * (y2 - y))
                + corners[1] * ((x - x1) * (y2 - y))
                + corners[2] * ((x2 - x) * (y - y1))
                + corners[3] * ((x - x1) * (y - y1)))
    };
    [
        do_interpolation_for_channel(0),
        do_interpolation_for_channel(1),
        do_interpolation_for_channel(2),
        do_interpolation_for_channel(3),
    ]
}

pub fn multiply_per_pixel(img1: &NDGrayImage, img2: &NDGrayImage) -> NDGrayImage {
    let img_shape = img1.shape();
    let mut out = Array3::zeros((img_shape[0], img_shape[1], 1));
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            out[[x, y, 0]] = img1[[x, y, 0]] * img2[[x, y, 0]];
        }
    }
    out
}

pub fn flip_vertically(img: &Array3<f64>) -> Array3<f64> {
    let img_shape = img.shape();
    let mut out = img.clone();
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            for channel in 0..img_shape[2] {
                out[[x, y, channel]] = img[[x, img_shape[1] - 1 - y, channel]];
            }
        }
    }
    out
}

pub fn invert(img: &Array3<f64>) -> Array3<f64> {
    let img_shape = img.shape();
    let mut out = img.clone();
    for x in 0..img_shape[0] {
        for y in 0..img_shape[1] {
            for channel in 0..img_shape[2] {
                out[[x, y, channel]] = 255.0 - img[[x, y, channel]];
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct Pointf {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct MyRgbaImage {
    pub nd_img: NDRgbaImage,
}

pub struct MyGrayImage {
    pub nd_img: NDGrayImage,
}

impl MyRgbaImage {
    pub fn set(&mut self, x: usize, y: usize, val: [f64; 4]) {
        self.nd_img[[x, y, 0]] = val[0];
        self.nd_img[[x, y, 1]] = val[1];
        self.nd_img[[x, y, 2]] = val[2];
        self.nd_img[[x, y, 3]] = val[3];
    }
    pub fn get(&self, x: usize, y: usize) -> [f64; 4] {
        [
            self.nd_img[[x, y, 0]],
            self.nd_img[[x, y, 1]],
            self.nd_img[[x, y, 2]],
            self.nd_img[[x, y, 3]],
        ]
    }
}

impl MyGrayImage {
    pub fn set(&mut self, x: usize, y: usize, val: [f64; 1]) {
        self.nd_img[[x, y, 0]] = val[0];
    }
    pub fn get(&self, x: usize, y: usize) -> [f64; 1] {
        [self.nd_img[[x, y, 0]]]
    }
}

impl From<(i64, i64)> for Point {
    fn from(coords: (i64, i64)) -> Point {
        Point {
            x: coords.0,
            y: coords.1,
            z: 0,
        }
    }
}

impl From<(i64, i64, i64)> for Point {
    fn from(coords: (i64, i64, i64)) -> Point {
        Point {
            x: coords.0,
            y: coords.1,
            z: coords.2,
        }
    }
}

impl From<(f64, f64)> for Pointf {
    fn from(coords: (f64, f64)) -> Pointf {
        Pointf {
            x: coords.0,
            y: coords.1,
            z: 0.0,
        }
    }
}

impl From<(f64, f64, f64)> for Pointf {
    fn from(coords: (f64, f64, f64)) -> Pointf {
        Pointf {
            x: coords.0,
            y: coords.1,
            z: coords.2,
        }
    }
}

impl From<(i64, i64)> for Pointf {
    fn from(coords: (i64, i64)) -> Pointf {
        Pointf {
            x: coords.0 as f64,
            y: coords.1 as f64,
            z: 0.0,
        }
    }
}

impl From<(i64, i64, i64)> for Pointf {
    fn from(coords: (i64, i64, i64)) -> Pointf {
        Pointf {
            x: coords.0 as f64,
            y: coords.1 as f64,
            z: coords.2 as f64,
        }
    }
}

impl From<Point> for Pointf {
    fn from(point: Point) -> Pointf {
        Pointf {
            x: point.x as f64,
            y: point.y as f64,
            z: point.z as f64,
        }
    }
}
