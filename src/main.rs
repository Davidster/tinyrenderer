mod helpers;
mod segment_3d;
mod model_renderer;

use anyhow::Result;
use nalgebra::{matrix, Vector3};
use segment_3d::Segment3D;
use show_image::WindowOptions;
use std::{
    cmp::Ordering, f64::consts::PI, fs::read_to_string, fs::File, io::BufReader, thread,
    time::Duration,
};

use helpers::*;
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

fn draw_triangle<'a>(
    img: &mut MyRgbaImage,
    z_buffer: &mut MyGrayImage,
    camera_pos: &nalgebra::Point3<f64>,
    time: i64,
    _texture_option: impl Into<Option<&'a MyRgbaImage>>,
    _normal_map_option: impl Into<Option<&'a MyRgbaImage>>,
    p1: MyTrianglePoint,
    p2: MyTrianglePoint,
    p3: MyTrianglePoint,
) -> Result<()> {
    let img_width = img.nd_img.shape()[0];
    let img_height = img.nd_img.shape()[1];
    let should_cull_point = |point: Pointf| {
        let _x = (point.x.round()) as i64;
        let _y = (point.y.round()) as i64;
        let dist = -point.z;
        _x < 0
            || _x >= img_width as i64
            || _y < 0
            || _y >= img_height as i64
            || dist > 1.0
            || dist < -1.0
    };

    let mut points = [p1, p2, p3].to_vec();

    if points.iter().all(|point| should_cull_point(point.position)) {
        return Ok(());
    }

    points.sort_by(|a, b| {
        b.position
            .y
            .partial_cmp(&a.position.y)
            .unwrap_or(Ordering::Equal)
    });
    // long means long in the y direction
    let long_side_segment = Segment3D::new(points[0].position, points[2].position);
    let long_side_split_point = long_side_segment.get_point(
        (points[1].position.y - points[0].position.y)
            / (points[2].position.y - points[0].position.y),
    )?;

    let mut total_res = 0;
    let texture_option = _texture_option.into();
    let normal_map_option = _normal_map_option.into();

    let t_bt_vectors = match (p1.color, p2.color, p3.color, normal_map_option) {
        (
            MyTrianglePointColor::Textured(uv1),
            MyTrianglePointColor::Textured(uv2),
            MyTrianglePointColor::Textured(uv3),
            Some(normal_map),
        ) => {
            let uv_mat = matrix![uv2.u - uv1.u, uv2.v - uv1.v;
                                 uv3.u - uv1.u, uv3.v - uv1.v;];
            let triangle_corner_edges = matrix![p2.position.x - p1.position.x, p2.position.y - p1.position.y, p2.position.z - p1.position.z;
                                                p3.position.x - p1.position.x, p3.position.y - p1.position.y, p3.position.z - p1.position.z;];
            match uv_mat.try_inverse() {
                Some(uv_mat_inv) => {
                    let tb_mat = uv_mat_inv * triangle_corner_edges;
                    let t_vector =
                        nalgebra::Vector3::new(tb_mat.m11, tb_mat.m12, tb_mat.m13).normalize();
                    let bt_vector =
                        nalgebra::Vector3::new(tb_mat.m21, tb_mat.m22, tb_mat.m23).normalize();
                    Some((uv1, uv2, uv3, t_vector, bt_vector, normal_map))
                }
                None => None,
            }
        }
        _ => None,
    };

    let mut fill_half_triangle = |segment_a: Segment3D,
                                  segment_b: Segment3D|
     -> anyhow::Result<()> {
        let resolution = ((segment_a.p2.x - segment_a.p1.x)
            .abs()
            .max((segment_a.p2.y - segment_a.p1.y).abs())
            .max((segment_b.p2.x - segment_a.p1.x).abs())
            .max((segment_b.p2.y - segment_a.p1.y).abs())
            .ceil() as i64)
            .max(2);
        total_res += resolution;
        let error_option = segment_a
            .get_point_iterator(resolution)?
            .zip(segment_b.get_point_iterator(resolution)?)
            .map(|(segment_a_point, segment_b_point)| -> anyhow::Result<()> {
                let horizontal_segment = Segment3D::new(segment_a_point, segment_b_point);
                horizontal_segment
                    .get_point_iterator(None)?
                    .for_each(|point| {
                        if should_cull_point(point) {
                            return;
                        }
                        let x = (point.x.round()) as usize;
                        let y = (point.y.round()) as usize;
                        let dist = -point.z;
                        let current_z = z_buffer.get(x, y)[0];
                        if current_z == f64::NEG_INFINITY || current_z > dist {
                            z_buffer.set(x, y, [dist]);
                            let (l1, l2, l3) = get_barycentric_coords_for_point_in_triangle(
                                (p1.position, p2.position, p3.position),
                                point,
                            );
                            let (n1, n2, n3) = (p1.normal, p2.normal, p3.normal);
                            // TODO: does this need to be normalized here?
                            let face_normal_vector = nalgebra::Vector3::new(
                                l1 * n1[0] + l2 * n2[0] + l3 * n3[0],
                                l1 * n1[1] + l2 * n2[1] + l3 * n3[1],
                                l1 * n1[2] + l2 * n2[2] + l3 * n3[2],
                            )
                            .normalize();

                            let normal_vector = match t_bt_vectors {
                                Some((uv1, uv2, uv3, t_vector, bt_vector, normal_map)) => {
                                    let u = l1 * uv1.u + l2 * uv2.u + l3 * uv3.u;
                                    let v = l1 * uv1.v + l2 * uv2.v + l3 * uv3.v;
                                    let texture_width = normal_map.nd_img.shape()[0];
                                    let texture_height = normal_map.nd_img.shape()[1];
                                    let normal_map_normal_rgb = sample_nd_img(
                                        &normal_map.nd_img,
                                        u * texture_width as f64,
                                        v * texture_height as f64,
                                    );
                                    let fix_rgb_range = |normal_coord_in_rgb_range| {
                                        ((normal_coord_in_rgb_range / 255.0) * 2.0) - 1.0
                                    };
                                    let normal_map_normal = nalgebra::Vector3::new(
                                        fix_rgb_range(normal_map_normal_rgb[0]),
                                        fix_rgb_range(normal_map_normal_rgb[1]),
                                        fix_rgb_range(normal_map_normal_rgb[2]),
                                    );
                                    let tbn_mat = nalgebra::Matrix3::new(
                                        t_vector.x,
                                        bt_vector.x,
                                        face_normal_vector.x,
                                        t_vector.y,
                                        bt_vector.y,
                                        face_normal_vector.y,
                                        t_vector.z,
                                        bt_vector.z,
                                        face_normal_vector.z,
                                    );
                                    (tbn_mat * normal_map_normal).normalize()
                                }

                                None => face_normal_vector,
                            };
                            let color_option = match (p1.color, p2.color, p3.color) {
                                (
                                    MyTrianglePointColor::Colored(c1),
                                    MyTrianglePointColor::Colored(c2),
                                    MyTrianglePointColor::Colored(c3),
                                ) => Some([
                                    l1 * c1[0] + l2 * c2[0] + l3 * c3[0],
                                    l1 * c1[1] + l2 * c2[1] + l3 * c3[1],
                                    l1 * c1[2] + l2 * c2[2] + l3 * c3[2],
                                    l1 * c1[3] + l2 * c2[3] + l3 * c3[3],
                                ]),
                                (
                                    MyTrianglePointColor::Textured(uv1),
                                    MyTrianglePointColor::Textured(uv2),
                                    MyTrianglePointColor::Textured(uv3),
                                ) => {
                                    let texture = texture_option
                                        .expect("Textured points were provided without a texture");
                                    let u = l1 * uv1.u + l2 * uv2.u + l3 * uv3.u;
                                    let v = l1 * uv1.v + l2 * uv2.v + l3 * uv3.v;
                                    let texture_width = texture.nd_img.shape()[0];
                                    let texture_height = texture.nd_img.shape()[1];
                                    Some(sample_nd_img(
                                        &texture.nd_img,
                                        u * texture_width as f64,
                                        v * texture_height as f64,
                                    ))
                                }
                                _ => None,
                            };
                            if let Some(albedo) = color_option {
                                // dbg!(color);
                                // let to_light_vec =
                                //     Vector3::new(0.1, 0.1, time as f64 * 0.000001).normalize();
                                let to_light_vec =
                                    Vector3::new(camera_pos.x, camera_pos.y, camera_pos.z)
                                        .normalize();
                                // let diffuse_proportion = face_normal_vector.dot(&to_light_vec).max(0.0);
                                // let diffuse_proportion = face_normal_vector.dot(&to_light_vec).abs();
                                let diffuse_proportion = normal_vector.dot(&to_light_vec).max(0.0);

                                let light_intensity = 1.0;
                                let ambient_light = 1.0;
                                let color = [
                                    (ambient_light
                                        + (diffuse_proportion * light_intensity * albedo[0]))
                                        .min(255.0),
                                    (ambient_light
                                        + (diffuse_proportion * light_intensity * albedo[1]))
                                        .min(255.0),
                                    (ambient_light
                                        + (diffuse_proportion * light_intensity * albedo[2]))
                                        .min(255.0),
                                    255.0,
                                ];
                                img.set(x, y, color);
                            }
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
        Segment3D::new(points[0].position, points[1].position),
        Segment3D::new(points[0].position, long_side_split_point),
    )?;
    fill_half_triangle(
        Segment3D::new(points[2].position, points[1].position),
        Segment3D::new(points[2].position, long_side_split_point),
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
    let _african_head_texture =
        flip_vertically(&load_nd_rgba_img_from_file("./src/african_head_diffuse.png").unwrap());
    let _african_head_normal_map =
        flip_vertically(&load_nd_rgba_img_from_file("./src/african_head_normal.png").unwrap());
    let african_head_texture = MyRgbaImage {
        nd_img: _african_head_texture,
    };
    let african_head_normal_map = MyRgbaImage {
        nd_img: _african_head_normal_map,
    };

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

    let mut triangle_index = 0;
    let mut time = 0;
    let mut showing_normal_map = false;
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
    //             // draw_triangle(
    //             //     &mut img,
    //             //     Pointf::from(transform_point((v1.x, v1.y))),
    //             //     Pointf::from(transform_point((v2.x, v2.y))),
    //             //     Pointf::from(transform_point((v3.x, v3.y))),
    //             //     colors[i % colors.len()],
    //             // )
    //             // .unwrap();

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
    //             draw(v1, v2, white);
    //             draw(v2, v3, white);
    //             draw(v3, v1, white);
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

    // loop {
    //     clear_screen(&mut img, &mut z_buffer);
    //     let mut _draw_triangle = |shape: &wavefront_obj::obj::Shape| {
    //         let prim = shape.primitive;
    //         if let wavefront_obj::obj::Primitive::Triangle(i1, i2, i3) = prim {
    //             // println!("Drawing triangle: {:?}", prim);
    //             let v1 = african_head_model.vertices[i1.0];
    //             let v2 = african_head_model.vertices[i2.0];
    //             let v3 = african_head_model.vertices[i3.0];
    //             let transform_point = |(x, y, z): (f64, f64, f64)| {
    //                 (
    //                     ((x + 1.0) * ((img_width - 1) as f64 / 2.0)),
    //                     ((y + 1.0) * ((img_height - 1) as f64 / 2.0)),
    //                     z,
    //                 )
    //             };

    //             let side_1 = Vector3::new(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z).normalize();
    //             let side_2 = Vector3::new(v3.x - v2.x, v3.y - v2.y, v3.z - v2.z).normalize();
    //             let normal = side_1.cross(&side_2).normalize();
    //             let to_light_vec = Vector3::new(0.1, 0.1, i as f64 * 0.000001).normalize();
    //             let diffuse_proportion = normal.dot(&to_light_vec);

    //             let light_intensity = 0.7;
    //             let diffuse_component = (diffuse_proportion * light_intensity * 255.0).max(0.0);
    //             let ambient_component = 30.0;
    //             let overall_light = (ambient_component + diffuse_component).min(255.0);
    //             let color = [overall_light, overall_light, overall_light, 255.0];
    //             let point_color = MyTrianglePointColor::Colored(color);
    //             draw_triangle(
    //                 &mut img,
    //                 &mut z_buffer,
    //                 MyTrianglePoint {
    //                     position: Pointf::from(transform_point((v1.x, v1.y, v1.z))),
    //                     color: point_color,
    //                 },
    //                 MyTrianglePoint {
    //                     position: Pointf::from(transform_point((v2.x, v2.y, v2.z))),
    //                     color: point_color,
    //                 },
    //                 MyTrianglePoint {
    //                     position: Pointf::from(transform_point((v3.x, v3.y, v3.z))),
    //                     color: point_color,
    //                 },
    //             )
    //             .unwrap();
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

    //     // window
    //     //     .set_image(
    //     //         "img",
    //     //         ndarray_to_image_gray(
    //     //             &flip_vertically(&get_drawable_z_buffer(&z_buffer.nd_img)),
    //     //             ImgConversionType::NORMALIZE,
    //     //         ),
    //     //     )
    //     //     .expect("Failed to set image");
    //     thread::sleep(Duration::from_millis(50));
    // }

    loop {
        time += 1;
        if time % 10 == 0 {
            showing_normal_map = !showing_normal_map;
        }
        clear_screen(&mut img, &mut z_buffer);
        let mut _draw_triangle = |shape: &wavefront_obj::obj::Shape| {
            let prim = shape.primitive;
            if let wavefront_obj::obj::Primitive::Triangle(i1, i2, i3) = prim {
                // let camera_direction = nalgebra::Vector3::new(0.04, 0.04, 0.1).normalize();
                let camera_direction = nalgebra::Vector3::new(0.0, 0.0, 1.0).normalize();
                let camera_direction_scaled = camera_direction * 2.0;
                let camera_pos = nalgebra::Point3::from(camera_direction_scaled);

                let camera_direction_matrix = nalgebra::Matrix4::look_at_rh(
                    &camera_pos,
                    &nalgebra::Point3::new(0.0, 0.0, 0.0),
                    &nalgebra::Vector3::new(0.0, 1.0, 0.0),
                );
                // let camera_direction_matrix = nalgebra::Matrix4::<f64>::identity();
                // dbg!(camera_direction_scaled);
                // let camera_translation_matrix = make_translation_matrix(-camera_direction_scaled);

                // let rotation_matrix = nalgebra::Matrix4::<f64>::identity();
                let rotation_matrix = make_rotation_matrix(0.04 * (time as f64), 0.0, 0.0);
                // let rotation_matrix = make_rotation_matrix(0.0, 0.0, 0.0);
                let translation_matrix =
                    make_translation_matrix(nalgebra::Vector3::new(0.0, 0.0, -0.004 * time as f64));
                // let translation_matrix =
                //     make_translation_matrix(camera_direction * (-0.004 * time as f64));

                let model_view_matrix =
                    camera_direction_matrix * translation_matrix * rotation_matrix;
                let horizontal_fov = PI / 2.0;
                let perspective_matrix = make_perspective_matrix(
                    10.0,
                    1.0,
                    horizontal_fov,
                    // horizontal_fov,
                    horizontal_fov * (img_height as f64 / img_width as f64),
                );

                let viewport_matrix =
                    make_scale_matrix(nalgebra::Vector3::new(
                        (img_width - 1) as f64 / 2.0,
                        (img_height - 1) as f64 / 2.0,
                        1.0,
                    )) * make_translation_matrix(nalgebra::Vector3::new(1.0, 1.0, 0.0));

                // Viewport * Projection * View * Model * vertex
                let transform_point = |local_point: (f64, f64, f64)| {
                    let global_point = viewport_matrix
                        * perspective_matrix
                        * model_view_matrix
                        * nalgebra::Vector4::new(local_point.0, local_point.1, local_point.2, 1.0);

                    let x = global_point.x / global_point.w;
                    let y = global_point.y / global_point.w;
                    let z = global_point.z / global_point.w;

                    (x, y, z)

                    // dbg!(x, y, z);

                    // let _z = z - 2.0;
                    // apply projection
                    // (
                    //     (((x + 1.0) / 2.0) * ((img_width - 1) as f64)),
                    //     (((y + 1.0) / 2.0) * ((img_height - 1) as f64)),
                    //     // (((x) + 1.0) * ((img_width - 1) as f64 / 2.0)),
                    //     // (((y) + 1.0) * ((img_height - 1) as f64 / 2.0)),
                    //     z,
                    // )
                };

                // let transform_point = |local_point: (f64, f64, f64)| {
                //     let global_point = model_view_matrix
                //         * nalgebra::Vector4::new(local_point.0, local_point.1, local_point.2, 1.0);

                //     let x = global_point.x;
                //     let y = global_point.y;
                //     let z = global_point.z;

                //     let _z = z - 2.0;
                //     // apply projection
                //     (
                //         (((x / -_z) + 1.0) * ((img_width - 1) as f64 / 2.0)),
                //         (((y / -_z) + 1.0) * ((img_height - 1) as f64 / 2.0)),
                //         // (((x) + 1.0) * ((img_width - 1) as f64 / 2.0)),
                //         // (((y) + 1.0) * ((img_height - 1) as f64 / 2.0)),
                //         _z,
                //     )
                // };

                let transform_normal = |local_normal: nalgebra::Vector3<f64>| {
                    model_view_matrix
                        .transpose()
                        .try_inverse()
                        .expect("Failed to transform normal")
                        * nalgebra::Vector4::new(
                            local_normal.x,
                            local_normal.y,
                            local_normal.z,
                            1.0,
                        )
                };

                // println!("Drawing triangle: {:?}", prim);
                let v1 = african_head_model.vertices[i1.0];
                let v2 = african_head_model.vertices[i2.0];
                let v3 = african_head_model.vertices[i3.0];

                let get_texture_for_vertex =
                    |(_, index_option, _): wavefront_obj::obj::VTNIndex| {
                        index_option.map(|index| african_head_model.tex_vertices[index])
                    };
                let uv1_option = get_texture_for_vertex(i1);
                let uv2_option = get_texture_for_vertex(i2);
                let uv3_option = get_texture_for_vertex(i3);

                let side_1 = Vector3::new(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z).normalize();
                let side_2 = Vector3::new(v3.x - v2.x, v3.y - v2.y, v3.z - v2.z).normalize();
                let face_normal = side_1.cross(&side_2).normalize();

                let get_normal_for_vertex = |(_, _, index_option): wavefront_obj::obj::VTNIndex| {
                    transform_normal(
                        index_option
                            .map(|index| african_head_model.normals[index])
                            .map(|normal| nalgebra::Vector3::new(normal.x, normal.y, normal.z))
                            .unwrap_or(face_normal),
                    )
                };
                let n1 = get_normal_for_vertex(i1);
                let n2 = get_normal_for_vertex(i2);
                let n3 = get_normal_for_vertex(i3);

                let point_color = MyTrianglePointColor::Colored(white);
                let triangle_colors: [MyTrianglePointColor; 3] =
                    match (uv1_option, uv2_option, uv3_option) {
                        (Some(uv1), Some(uv2), Some(uv3)) => [
                            MyTrianglePointColor::Textured(TextureCoords { u: uv1.u, v: uv1.v }),
                            MyTrianglePointColor::Textured(TextureCoords { u: uv2.u, v: uv2.v }),
                            MyTrianglePointColor::Textured(TextureCoords { u: uv3.u, v: uv3.v }),
                        ],
                        _ => [point_color, point_color, point_color],
                    };

                draw_triangle(
                    &mut img,
                    &mut z_buffer,
                    &camera_pos,
                    time,
                    &african_head_texture,
                    // Some(&african_head_normal_map),
                    if showing_normal_map {
                        Some(&african_head_normal_map)
                    } else {
                        None
                    },
                    MyTrianglePoint {
                        position: Pointf::from(transform_point((v1.x, v1.y, v1.z))),
                        // color: point_color,
                        color: triangle_colors[0],
                        normal: n1,
                        // normal: face_normal,
                    },
                    MyTrianglePoint {
                        position: Pointf::from(transform_point((v2.x, v2.y, v2.z))),
                        // color: point_color,
                        color: triangle_colors[1],
                        normal: n2,
                        // normal: face_normal,
                    },
                    MyTrianglePoint {
                        position: Pointf::from(transform_point((v3.x, v3.y, v3.z))),
                        // color: point_color,
                        color: triangle_colors[2],
                        normal: n3,
                        // normal: face_normal,
                    },
                )
                .unwrap();
                triangle_index += 1;
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

    // let color = sample_nd_img(&african_head_texture.nd_img, 0.0, 0.0);
    // dbg!(color);
    // let camera_matrix_ = nalgebra::Matrix4::look_at_rh(
    //     &nalgebra::Point3::new(0.0, 0.0, 1.0),
    //     &nalgebra::Point3::new(0.0, 0.0, 0.0),
    //     &nalgebra::Vector3::new(0.0, 1.0, 0.0),
    // );
    // draw_triangle(
    //     &mut img,
    //     &mut z_buffer,
    //     &nalgebra::Point3::new(0.0, 0.0, 1.0),
    //     0,
    //     &african_head_texture,
    //     None,
    //     MyTrianglePoint {
    //         position: Pointf::from((img_width as f64 * 0.5, img_height as f64 * 0.75)),
    //         // color: MyTrianglePointColor::Colored(color),
    //         color: MyTrianglePointColor::Textured(TextureCoords {
    //             u: img_width as f64 * 0.5,
    //             v: img_height as f64 * 0.75,
    //         }),
    //         normal: nalgebra::Vector4::new(0.0, 0.0, 1.0, 0.0),
    //     },
    //     MyTrianglePoint {
    //         position: Pointf::from((img_width as f64 * 0.75, img_height as f64 * 0.25)),
    //         // color: MyTrianglePointColor::Colored(color),
    //         color: MyTrianglePointColor::Textured(TextureCoords {
    //             u: img_width as f64 * 0.75,
    //             v: img_height as f64 * 0.25,
    //         }),
    //         normal: nalgebra::Vector4::new(0.0, 0.0, 1.0, 0.0),
    //     },
    //     MyTrianglePoint {
    //         position: Pointf::from((img_width as f64 * 0.25, img_height as f64 * 0.25)),
    //         // color: MyTrianglePointColor::Colored(color),
    //         color: MyTrianglePointColor::Textured(TextureCoords {
    //             u: img_width as f64 * 0.25,
    //             v: img_height as f64 * 0.25,
    //         }),
    //         normal: nalgebra::Vector4::new(0.0, 0.0, 1.0, 0.0),
    //     },
    // )
    // .unwrap();
    // window
    //     .set_image("img", ndarray_to_image_rgba(&flip_vertically(&img.nd_img)))
    //     .expect("Failed to set image");

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
