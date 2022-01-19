mod helpers;
mod segment_3d;

use anyhow::Result;
use nalgebra::Vector3;
use segment_3d::Segment3D;
use show_image::WindowOptions;
use std::{cmp::Ordering, fs::read_to_string, fs::File, io::BufReader, thread, time::Duration};

use helpers::{
    flip_vertically, invert, ndarray_to_image_gray, ndarray_to_image_rgba,
    wait_for_windows_to_close, write_nd_rgba_img_to_file, ImgConversionType, MyGrayImage,
    MyRgbaImage, NDGrayImage, NDRgbaImage, Point, Pointf,
};
use ndarray::Array3;

const black: [f64; 4] = [0.0, 0.0, 0.0, 255.0];
const red: [f64; 4] = [255.0, 0.0, 0.0, 255.0];
const green: [f64; 4] = [0.0, 255.0, 0.0, 255.0];
const blue: [f64; 4] = [0.0, 0.0, 255.0, 255.0];
const white: [f64; 4] = [255.0, 255.0, 255.0, 255.0];

fn draw_line(
    img: &mut MyRgbaImage,
    start_point: Pointf,
    end_point: Pointf,
    color: [f64; 4],
) -> Result<()> {
    let mut points = [start_point, end_point].to_vec();
    points.sort_by(|a, b| b.x.partial_cmp(&a.x).unwrap_or(Ordering::Equal));
    let segment = Segment3D::new(points[0], points[1]);
    // println!("Created segment: {:?}", segment);
    segment.get_point_iterator(None)?.for_each(|point| {
        img.set(
            (point.x.round()) as usize,
            (point.y.round()) as usize,
            color,
        );
    });
    Ok(())
}

fn draw_triangle(
    img: &mut MyRgbaImage,
    z_buffer: &mut MyGrayImage,
    p1: Pointf,
    p2: Pointf,
    p3: Pointf,
    color: [f64; 4],
) -> Result<()> {
    // println!("drawing triangle: {:?}, {:?}, {:?}", p1, p2, p3);
    let mut points = [p1, p2, p3].to_vec();
    points.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(Ordering::Equal));
    // long means long in the y direction
    let long_side_segment = Segment3D::new(points[0], points[2]);
    let long_side_split_point =
        long_side_segment.get_point((points[1].y - points[0].y) / (points[2].y - points[0].y))?;

    let mut fill_half_triangle =
        |segment_a: Segment3D, segment_b: Segment3D| -> anyhow::Result<()> {
            let resolution = ((segment_a.p2.x - segment_a.p1.x)
                .abs()
                .max((segment_a.p2.y - segment_a.p1.y).abs())
                .max((segment_b.p2.x - segment_a.p1.x).abs())
                .max((segment_b.p2.y - segment_a.p1.y).abs())
                .ceil() as i64)
                .max(2);
            let error_option = segment_a
                .get_point_iterator(resolution)?
                .zip(segment_b.get_point_iterator(resolution)?)
                .map(|(p1, p2)| -> anyhow::Result<()> {
                    let horizontal_segment = Segment3D::new(p1, p2);
                    horizontal_segment
                        .get_point_iterator(None)?
                        .for_each(|point| {
                            let x = (point.x.round()) as usize;
                            let y = (point.y.round()) as usize;
                            let dist = -point.z + 1.0;
                            let current_z = z_buffer.get(x, y)[0];
                            if current_z == f64::NEG_INFINITY || current_z > dist {
                                z_buffer.set(x, y, [dist]);
                                img.set(x, y, color);
                            }
                        });
                    Ok(())
                })
                .find_map(|result| -> Option<anyhow::Result<()>> {
                    match result {
                        Ok(_) => None,
                        Err(err) => Some(Err(err)),
                    }
                });

            if let Some(result) = error_option {
                return result;
            }
            Ok(())
        };

    fill_half_triangle(
        Segment3D::new(points[0], points[1]),
        Segment3D::new(points[0], long_side_split_point),
    )?;
    fill_half_triangle(
        Segment3D::new(points[2], points[1]),
        Segment3D::new(points[2], long_side_split_point),
    )?;

    Ok(())
}

fn load_model(path: &str) -> Result<wavefront_obj::obj::Object> {
    let mut objects = wavefront_obj::obj::parse(read_to_string(path)?)?.objects;
    Ok(objects.remove(0))
}

fn clear_screen(img: &mut MyRgbaImage, z_buffer: &mut MyGrayImage) {
    let img_width = img.nd_img.shape()[0];
    let img_height = img.nd_img.shape()[1];
    // set all to black
    for x in 0..img_width {
        for y in 0..img_height {
            img.set(x, y, black.clone());
        }
    }
    // reset z-buffer
    for x in 0..img_width {
        for y in 0..img_height {
            z_buffer.set(x, y, [f64::NEG_INFINITY]);
        }
    }
}

fn get_drawable_z_buffer(z_buffer: &NDGrayImage) -> NDGrayImage {
    let width = z_buffer.shape()[0];
    let height = z_buffer.shape()[1];
    let mut drawable_z_buffer: NDGrayImage = Array3::zeros((width, height, 1));
    for x in 0..width {
        for y in 0..height {
            let val = z_buffer[[x, y, 0]].max(0.0);
            if val > 0.0 {
                drawable_z_buffer[[x, y, 0]] = 1.0 / val;
            }
        }
    }
    drawable_z_buffer
}

#[show_image::main]
fn main() {
    let mut img = MyRgbaImage {
        nd_img: Array3::zeros((1000, 1000, 4)),
    };
    let img_width = img.nd_img.shape()[0];
    let img_height = img.nd_img.shape()[1];
    let mut z_buffer = MyGrayImage {
        nd_img: Array3::zeros((img_width, img_height, 1)),
    };

    clear_screen(&mut img, &mut z_buffer);

    // img.set(52, 41, [255.0, 0.0, 0.0, 255.0]);
    // write_nd_rgba_img_to_file("./1.png", &flip_vertically(&img.nd_img)).unwrap();
    // draw_line_2(&mut img, Point::from((20, 13)), Point::from((40, 80)), red);
    // write_nd_rgba_img_to_file("./2.png", &flip_vertically(&img.nd_img)).unwrap();
    // draw_line_2(
    //     &mut img,
    //     Pointf::from((80, 40)),
    //     Pointf::from((13, 20)),
    //     red,
    // )
    // .unwrap();
    // write_nd_rgba_img_to_file("./3.png", &flip_vertically(&img.nd_img)).unwrap();
    let african_head_model = load_model("./src/african_head.obj").unwrap();

    let window = show_image::create_window(
        "img",
        WindowOptions::new().set_size([img_width as u32, img_height as u32]),
    )
    .expect("Failed to create window");
    // window
    //     .set_image("img", ndarray_to_image_rgba(&flip_vertically(&img.nd_img)))
    //     .expect("Failed to set image");

    // let mut line_length = 20.0f64;
    // let mut line_angle = 0.0f64;
    // let center = Point::from((img_width as i64 / 2, img_height as i64 / 2));
    // loop {
    //     for x in 0..img_width {
    //         for y in 0..img_height {
    //             img.set(x, y, black.clone());
    //         }
    //     }
    //     let line_2_angle = line_angle + 0.5;
    //     draw_line_2(
    //         &mut img,
    //         Pointf::from(center),
    //         Pointf::from((
    //             center.x as f64 + (line_2_angle.cos() * line_length).round(),
    //             center.y as f64 + (line_2_angle.sin() * line_length).round(),
    //         )),
    //         red,
    //     )
    //     .expect("Failed to draw line");
    //     window
    //         .set_image("img", ndarray_to_image_rgba(&flip_vertically(&img.nd_img)))
    //         .expect("Failed to set image");
    //     line_angle += 0.1;
    //     // line_length += 0.5;
    //     thread::sleep(Duration::from_millis(50));
    // }

    // println!("{:?}", african_head_model);
    // let mut iter = african_head_model.indices.iter();
    // iter.next();
    // african_head_model
    //     .indices
    //     .iter()
    //     .zip(iter)
    //     .for_each(|(i1, i2)| {
    //         let v1 = african_head_model.vertices[*i1 as usize];
    //         let v2 = african_head_model.vertices[*i2 as usize];
    //         // dbg!(v1.position[0] + 1.0);
    //         // dbg!((img_width - 1) as f32 / 2.0);
    //         // dbg!(((v1.position[0] + 1.0) * ((img_width - 1) as f32 / 2.0)).round() as usize);
    //         draw_line(
    // &mut img,
    // Point::from((
    //     ((v1.position[0] + 1.0) * ((img_width - 1) as f32 / 2.0)).round() as usize,
    //     ((v1.position[1] + 1.0) * ((img_height - 1) as f32 / 2.0)).round() as usize,
    // )),
    // Point::from((
    //     ((v2.position[0] + 1.0) * ((img_width - 1) as f32 / 2.0)).round() as usize,
    //     ((v2.position[1] + 1.0) * ((img_height - 1) as f32 / 2.0)).round() as usize,
    // )),
    // white,
    //         );
    //     });

    let mut i = 0;
    let colors = [red, blue, green];

    // _draw_triangle(&african_head_model.geometry[0].shapes[50]);
    // _draw_triangle(&african_head_model.geometry[0].shapes[51]);
    // _draw_triangle(&african_head_model.geometry[0].shapes[52]);
    // _draw_triangle(&african_head_model.geometry[0].shapes[53]);
    // _draw_triangle(&african_head_model.geometry[0].shapes[54]);
    // _draw_triangle(&african_head_model.geometry[0].shapes[55]);
    // _draw_triangle(&african_head_model.geometry[0].shapes[56]);
    // _draw_triangle(&african_head_model.geometry[0].shapes[57]);

    // loop {
    //     for x in 0..img_width {
    //         for y in 0..img_height {
    //             img.set(x, y, black.clone());
    //         }
    //     }
    //     let mut _draw_triangle = |shape: &wavefront_obj::obj::Shape| {
    //         let prim = shape.primitive;
    //         if let wavefront_obj::obj::Primitive::Triangle(i1, i2, i3) = prim {
    //             // println!("Drawing triangle: {:?}", prim);
    //             let v1 = african_head_model.vertices[i1.0];
    //             let v2 = african_head_model.vertices[i2.0];
    //             let v3 = african_head_model.vertices[i3.0];
    //             let transform_point = |(x, y): (f64, f64)| {
    //                 (
    //                     ((x + 1.0) * ((img_width - 1) as f64 / 2.0)) as i64,
    //                     ((y + 1.0) * ((img_height - 1) as f64 / 2.0)) as i64,
    //                 )
    //             };
    //             draw_triangle(
    //                 &mut img,
    //                 Pointf::from(transform_point((v1.x, v1.y))),
    //                 Pointf::from(transform_point((v2.x, v2.y))),
    //                 Pointf::from(transform_point((v3.x, v3.y))),
    //                 colors[i % colors.len()],
    //             )
    //             .unwrap();

    //             let mut draw = |v_1: wavefront_obj::obj::Vertex,
    //                             v_2: wavefront_obj::obj::Vertex,
    //                             color: [f64; 4]| {
    //                 draw_line(
    //                     &mut img,
    //                     Pointf::from((
    //                         ((v_1.x + 1.0) * ((img_width - 1) as f64 / 2.0)).round(),
    //                         ((v_1.y + 1.0) * ((img_height - 1) as f64 / 2.0)).round(),
    //                     )),
    //                     Pointf::from((
    //                         ((v_2.x + 1.0) * ((img_width - 1) as f64 / 2.0)).round(),
    //                         ((v_2.y + 1.0) * ((img_height - 1) as f64 / 2.0)).round(),
    //                     )),
    //                     color,
    //                 )
    //                 .unwrap();
    //             };
    //             // draw(v1, v2, white);
    //             // draw(v2, v3, white);
    //             // draw(v3, v1, white);
    //             i += 1;
    //         }
    //     };
    //     african_head_model.geometry[0]
    //         .shapes
    //         .iter()
    //         .for_each(_draw_triangle);
    //     window
    //         .set_image("img", ndarray_to_image_rgba(&flip_vertically(&img.nd_img)))
    //         .expect("Failed to set image");
    //     thread::sleep(Duration::from_millis(50));
    // }

    loop {
        clear_screen(&mut img, &mut z_buffer);
        let mut _draw_triangle = |shape: &wavefront_obj::obj::Shape| {
            let prim = shape.primitive;
            if let wavefront_obj::obj::Primitive::Triangle(i1, i2, i3) = prim {
                // println!("Drawing triangle: {:?}", prim);
                let v1 = african_head_model.vertices[i1.0];
                let v2 = african_head_model.vertices[i2.0];
                let v3 = african_head_model.vertices[i3.0];
                let transform_point = |(x, y, z): (f64, f64, f64)| {
                    (
                        ((x + 1.0) * ((img_width - 1) as f64 / 2.0)),
                        ((y + 1.0) * ((img_height - 1) as f64 / 2.0)),
                        z,
                    )
                };

                let side_1 = Vector3::new(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z).normalize();
                let side_2 = Vector3::new(v3.x - v2.x, v3.y - v2.y, v3.z - v2.z).normalize();
                let normal = side_1.cross(&side_2).normalize();
                let to_light_vec = Vector3::new(0.1, 0.1, i as f64 * 0.000001).normalize();
                let diffuse_proportion = normal.dot(&to_light_vec);

                let light_intensity = 0.7;
                let diffuse_component = (diffuse_proportion * light_intensity * 255.0).max(0.0);
                let ambient_component = 30.0;
                let overall_light = (ambient_component + diffuse_component).min(255.0);
                draw_triangle(
                    &mut img,
                    &mut z_buffer,
                    Pointf::from(transform_point((v1.x, v1.y, v1.z))),
                    Pointf::from(transform_point((v2.x, v2.y, v2.z))),
                    Pointf::from(transform_point((v3.x, v3.y, v3.z))),
                    [overall_light, overall_light, overall_light, 255.0],
                )
                .unwrap();
                i += 1;
            }
        };
        african_head_model.geometry[0]
            .shapes
            .iter()
            .for_each(_draw_triangle);
        window
            .set_image("img", ndarray_to_image_rgba(&flip_vertically(&img.nd_img)))
            .expect("Failed to set image");

        // window
        //     .set_image(
        //         "img",
        //         ndarray_to_image_gray(
        //             &flip_vertically(&get_drawable_z_buffer(&z_buffer.nd_img)),
        //             ImgConversionType::NORMALIZE,
        //         ),
        //     )
        //     .expect("Failed to set image");
        thread::sleep(Duration::from_millis(50));
    }

    // african_head_model.geometry[0]
    //     .shapes
    //     .iter()
    //     .for_each(_draw_triangle);
    // write_nd_rgba_img_to_file("./face.png", &flip_vertically(&img.nd_img)).unwrap();
    // window
    //     .set_image("img", ndarray_to_image_rgba(&flip_vertically(&img.nd_img)))
    //     .expect("Failed to set image");
    // wait_for_windows_to_close([window].to_vec()).expect("Failed to wait on windows");
}

// old version
// fn draw_triangle(img: &mut MyRgbaImage, p1: Point, p2: Point, p3: Point, color: [f64; 4]) {
//     println!("drawing triangle: {:?}, {:?}, {:?}", p1, p2, p3);
//     let mut points = [p1, p2, p3].to_vec();
//     // draw half triangle from p_a to p_b
//     let mut draw_half_triangle = |points: &Vec<Point>, downwards: bool| {
//         println!(
//             "drawing half triangle: {:?}, {:?}, {:?}",
//             points[0], points[1], points[2]
//         );
//         // line one
//         let d_x_1 = if downwards {
//             (points[1].x - points[0].x) as f64
//         } else {
//             (points[1].x - points[2].x) as f64
//         };

//         let d_y = if downwards {
//             (points[1].y - points[0].y) as f64
//         } else {
//             (points[1].y - points[2].y) as f64
//         };

//         // line two
//         let d_x_2 = if downwards {
//             (points[2].x - points[0].x) as f64
//         } else {
//             (points[0].x - points[2].x) as f64
//         };

//         let horizontal_lines = d_y.abs() as i64;

//         (0..horizontal_lines).for_each(|i| {
//             let edge_1_x = if downwards {
//                 points[0].x + (i as f64 * (d_x_1 as f64 / horizontal_lines as f64)).round() as i64
//             } else {
//                 points[2].x + (i as f64 * (d_x_1 as f64 / horizontal_lines as f64)).round() as i64
//             };

//             let edge_2_x = if downwards {
//                 points[0].x + (i as f64 * (d_x_2 as f64 / horizontal_lines as f64)).round() as i64
//             } else {
//                 points[2].x + (i as f64 * (d_x_2 as f64 / horizontal_lines as f64)).round() as i64
//             };
//             // let edge_1_x =
//             //     points[0].x + (i as f64 * (d_x_1 as f64 / horizontal_lines as f64)).round() as i64;
//             // let edge_2_x =
//             //     points[0].x + (i as f64 * (d_x_2 as f64 / horizontal_lines as f64)).round() as i64;
//             let y = if downwards {
//                 points[0].y - i
//             } else {
//                 points[2].y + i
//             };
//             let edge_range = if edge_1_x < edge_2_x {
//                 println!(
//                     "drawing line from {:?} to {:?} on y={:?}",
//                     edge_1_x, edge_2_x, y
//                 );
//                 edge_1_x..(edge_2_x + 1)
//             } else {
//                 println!(
//                     "drawing line from {:?} to {:?} on y={:?}",
//                     edge_2_x, edge_1_x, y
//                 );
//                 edge_2_x..(edge_1_x + 1)
//             };

//             edge_range.for_each(|x| {
//                 println!("drawing point: {:?}, {:?}", x, y);
//                 img.set(x as usize, y as usize, color);
//             });
//         });
//     };
//     points.sort_by(|a, b| b.y.cmp(&a.y));
//     draw_half_triangle(&points, true);
//     // points.sort_by(|a, b| a.y.cmp(&b.y));
//     // draw_half_triangle(&points, false);
// }

// old version
// fn draw_line(img: &mut MyRgbaImage, _start_point: Point, _end_point: Point, color: [f64; 4]) {
//     let d_x_a = (_end_point.x - _start_point.x).abs();
//     let d_y_a = (_end_point.y - _start_point.y).abs();
//     let points = d_x_a.max(d_y_a);

//     let (start_point, end_point) = if _start_point.x < _end_point.x {
//         (_start_point, _end_point)
//     } else {
//         (_end_point, _start_point)
//     };
//     // println!("drawing line from {:?} to {:?}", start_point, end_point);

//     // dbg!(start_point);
//     // dbg!(end_point);
//     // dbg!(along_x);
//     // let slope = (end_point.y - start_point.y) as f64 / (end_point.x - start_point.x) as f64;
//     let d_x = (end_point.x - start_point.x) as f64;
//     let d_y = (end_point.y - start_point.y) as f64;
//     let d_x_n = d_x as f64 / points as f64;
//     let d_y_n = d_y as f64 / points as f64;

//     (0..points).for_each(|i| {
//         // let t_x = i as f64 * d_x_n;
//         // let t_y = i as f64 * d_y_n;
//         img.set(
//             (start_point.x + (i as f64 * d_x_n).round() as i64) as usize,
//             (start_point.y + (i as f64 * d_y_n).round() as i64) as usize,
//             color,
//         );
//     });
//     // (start_point.x..(end_point.x + 1))
//     //     .enumerate()
//     //     .for_each(|(t, x)| {
//     //         // println!(
//     //         //     "drawing point at {:?}, {:?}",
//     //         //     x,
//     //         //     start_point.y as usize + (t as f64 * slope).round() as usize
//     //         // );
//     //         img.set(
//     //             x as usize,
//     //             start_point.y as usize + (t as f64 * slope).round() as usize,
//     //             color,
//     //         );
//     //     });
// }
