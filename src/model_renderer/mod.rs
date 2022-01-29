use super::helpers::*;
use super::segment_3d::*;

use std::marker::PhantomData;

pub struct ModelRendererState {
    pub frame_buffer: MyRgbaImage,
    pub z_buffer: MyGrayImage,
    pub texture: Option<MyRgbaImage>,
    pub normal_map: Option<MyRgbaImage>,
    pub model: wavefront_obj::obj::Object,
    // phantom: PhantomData<'a>,
}

pub fn render_model<
    // Option<nalgebra::Vector3<f64>, Option<wavefront_obj::obj::TVertex>
    V: FnMut(VertexShaderArgs) -> VertexShaderResult,
    F: FnMut(FragmentShaderArgs) -> FragmentShaderResult,
>(
    model_renderer_state: &mut ModelRendererState,
    vertex_shader: &mut V,
    fragment_shader: &mut F,
) {
    let frame_width = model_renderer_state.frame_buffer.nd_img.shape()[0];
    let frame_height = model_renderer_state.frame_buffer.nd_img.shape()[1];
    let texture_option = &model_renderer_state.texture;
    let normal_map_option = &model_renderer_state.normal_map;
    let frame_buffer = &mut model_renderer_state.frame_buffer;
    let z_buffer = &mut model_renderer_state.z_buffer;
    let model = &model_renderer_state.model;
    let shapes = &model.geometry[0].shapes;
    let vertex_shader_results: Vec<
        Option<(VertexShaderResult, VertexShaderResult, VertexShaderResult)>,
    > = shapes
        .iter()
        .map(|shape| {
            match shape.primitive {
                wavefront_obj::obj::Primitive::Triangle(i1, i2, i3) => {
                    // get triangle vertices and run them through the vertex shader
                    let v1 = model.vertices[i1.0];
                    let v2 = model.vertices[i2.0];
                    let v3 = model.vertices[i3.0];
                    let get_face_normal = || {
                        let side_1 = nalgebra::Vector3::new(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z)
                            .normalize();
                        let side_2 = nalgebra::Vector3::new(v3.x - v2.x, v3.y - v2.y, v3.z - v2.z)
                            .normalize();
                        let face_normal = side_1.cross(&side_2).normalize();
                        face_normal
                    };
                    let get_normal_for_vertex =
                        |(_, _, index_option): wavefront_obj::obj::VTNIndex| {
                            index_option
                                .map(|index| model.normals[index])
                                .map(|normal| nalgebra::Vector3::new(normal.x, normal.y, normal.z))
                                .unwrap_or_else(get_face_normal)
                        };

                    let n1 = get_normal_for_vertex(i1);
                    let n2 = get_normal_for_vertex(i2);
                    let n3 = get_normal_for_vertex(i3);

                    let get_texture_for_vertex =
                        |(_, index_option, _): wavefront_obj::obj::VTNIndex| {
                            index_option.map(|index| model.tex_vertices[index])
                        };
                    let uv1_option = get_texture_for_vertex(i1);
                    let uv2_option = get_texture_for_vertex(i2);
                    let uv3_option = get_texture_for_vertex(i3);

                    Some((
                        vertex_shader(VertexShaderArgs::from((v1, n1, uv1_option))),
                        vertex_shader(VertexShaderArgs::from((v2, n2, uv2_option))),
                        vertex_shader(VertexShaderArgs::from((v3, n3, uv3_option))),
                    ))
                }
                _ => None,
            }
        })
        .collect();
    let error_option = vertex_shader_results
        .iter()
        .flatten()
        .map(|(vsr1, vsr2, vsr3)| {
            let should_cull_point = |point: &nalgebra::Vector3<f64>| {
                point.x < -1.0
                    || point.x >= 1.0
                    || point.y < -1.0
                    || point.y >= 1.0
                    || point.z < -1.0
                    || point.z > 1.0
            };

            let mut points = [
                vsr1.clip_space_position,
                vsr2.clip_space_position,
                vsr3.clip_space_position,
            ]
            .to_vec();

            if points.iter().all(|point| should_cull_point(point)) {
                return Ok(());
            }

            let viewport_matrix =
                make_scale_matrix(nalgebra::Vector3::new(
                    (frame_width - 1) as f64 / 2.0,
                    (frame_height - 1) as f64 / 2.0,
                    1.0,
                )) * make_translation_matrix(nalgebra::Vector3::new(1.0, 1.0, 0.0));

            let mut viewport_points: Vec<nalgebra::Vector4<f64>> = points
                .iter()
                .map(|point| {
                    viewport_matrix * nalgebra::Vector4::new(point.x, point.y, point.z, 1.0)
                })
                // TODO: is this needed?
                .map(|point| {
                    nalgebra::Vector4::new(
                        point.x / point.w,
                        point.y / point.w,
                        point.z / point.w,
                        1.0,
                    )
                })
                .collect();

            let viewport_points_unsorted = viewport_points.clone();

            viewport_points
                .sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal));

            // TODO: move into shader code
            // let t_bt_vectors = match (p1.color, p2.color, p3.color, normal_map_option) {
            //     (
            //         MyTrianglePointColor::Textured(uv1),
            //         MyTrianglePointColor::Textured(uv2),
            //         MyTrianglePointColor::Textured(uv3),
            //         Some(normal_map),
            //     ) => {
            //         let uv_mat = matrix![uv2.u - uv1.u, uv2.v - uv1.v;
            //                                 uv3.u - uv1.u, uv3.v - uv1.v;];
            //         let triangle_corner_edges = matrix![p2.position.x - p1.position.x, p2.position.y - p1.position.y, p2.position.z - p1.position.z;
            //                                             p3.position.x - p1.position.x, p3.position.y - p1.position.y, p3.position.z - p1.position.z;];
            //         match uv_mat.try_inverse() {
            //             Some(uv_mat_inv) => {
            //                 let tb_mat = uv_mat_inv * triangle_corner_edges;
            //                 let t_vector =
            //                     nalgebra::Vector3::new(tb_mat.m11, tb_mat.m12, tb_mat.m13).normalize();
            //                 let bt_vector =
            //                     nalgebra::Vector3::new(tb_mat.m21, tb_mat.m22, tb_mat.m23).normalize();
            //                 Some((uv1, uv2, uv3, t_vector, bt_vector, normal_map))
            //             }
            //             None => None,
            //         }
            //     }
            //     _ => None,
            // };

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
                let error_option = segment_a
                    .get_point_iterator(resolution)?
                    .zip(segment_b.get_point_iterator(resolution)?)
                    .map(|(segment_a_point, segment_b_point)| -> anyhow::Result<()> {
                        let horizontal_segment = Segment3D::new(segment_a_point, segment_b_point);
                        horizontal_segment.get_point_iterator(None)?.for_each(
                            |viewport_space_position| {
                                if should_cull_point(&viewport_space_position) {
                                    return;
                                }
                                let x = (viewport_space_position.x.round()) as usize;
                                let y = (viewport_space_position.y.round()) as usize;
                                let dist = -viewport_space_position.z;
                                let current_z = z_buffer.get(x, y)[0];
                                if current_z == f64::NEG_INFINITY || current_z > dist {
                                    z_buffer.set(x, y, [dist]);
                                    let barycentric_coords =
                                        get_barycentric_coords_for_point_in_triangle(
                                            (
                                                viewport_points_unsorted[0],
                                                viewport_points_unsorted[1],
                                                viewport_points_unsorted[2],
                                            ),
                                            viewport_space_position,
                                        );
                                    let (l1, l2, l3) = barycentric_coords;
                                    let (n1, n2, n3) = (vsr1.normal, vsr2.normal, vsr3.normal);
                                    // TODO: does this need to be normalized here?
                                    let normal_interp = nalgebra::Vector3::new(
                                        l1 * n1[0] + l2 * n2[0] + l3 * n3[0],
                                        l1 * n1[1] + l2 * n2[1] + l3 * n3[1],
                                        l1 * n1[2] + l2 * n2[2] + l3 * n3[2],
                                    )
                                    .normalize();

                                    // TODO: move into shader code
                                    // let normal_vector = match t_bt_vectors {
                                    //     Some((uv1, uv2, uv3, t_vector, bt_vector, normal_map)) => {
                                    //         let u = l1 * uv1.u + l2 * uv2.u + l3 * uv3.u;
                                    //         let v = l1 * uv1.v + l2 * uv2.v + l3 * uv3.v;
                                    //         let texture_width = normal_map.nd_img.shape()[0];
                                    //         let texture_height = normal_map.nd_img.shape()[1];
                                    //         let normal_map_normal_rgb = sample_nd_img(
                                    //             &normal_map.nd_img,
                                    //             u * texture_width as f64,
                                    //             v * texture_height as f64,
                                    //         );
                                    //         let fix_rgb_range = |normal_coord_in_rgb_range| {
                                    //             ((normal_coord_in_rgb_range / 255.0) * 2.0) - 1.0
                                    //         };
                                    //         let normal_map_normal = nalgebra::Vector3::new(
                                    //             fix_rgb_range(normal_map_normal_rgb[0]),
                                    //             fix_rgb_range(normal_map_normal_rgb[1]),
                                    //             fix_rgb_range(normal_map_normal_rgb[2]),
                                    //         );
                                    //         let tbn_mat = nalgebra::Matrix3::new(
                                    //             t_vector.x,
                                    //             bt_vector.x,
                                    //             face_normal_vector.x,
                                    //             t_vector.y,
                                    //             bt_vector.y,
                                    //             face_normal_vector.y,
                                    //             t_vector.z,
                                    //             bt_vector.z,
                                    //             face_normal_vector.z,
                                    //         );
                                    //         (tbn_mat * normal_map_normal).normalize()
                                    //     }

                                    //     None => face_normal_vector,
                                    // };

                                    // TODO: move into shader code
                                    // let color_option = match (p1.color, p2.color, p3.color) {
                                    //     (
                                    //         MyTrianglePointColor::Colored(c1),
                                    //         MyTrianglePointColor::Colored(c2),
                                    //         MyTrianglePointColor::Colored(c3),
                                    //     ) => Some([
                                    //         l1 * c1[0] + l2 * c2[0] + l3 * c3[0],
                                    //         l1 * c1[1] + l2 * c2[1] + l3 * c3[1],
                                    //         l1 * c1[2] + l2 * c2[2] + l3 * c3[2],
                                    //         l1 * c1[3] + l2 * c2[3] + l3 * c3[3],
                                    //     ]),
                                    //     (
                                    //         MyTrianglePointColor::Textured(uv1),
                                    //         MyTrianglePointColor::Textured(uv2),
                                    //         MyTrianglePointColor::Textured(uv3),
                                    //     ) => {
                                    //         let texture = texture_option.expect(
                                    //             "Textured points were provided without a texture",
                                    //         );
                                    //         let u = l1 * uv1.u + l2 * uv2.u + l3 * uv3.u;
                                    //         let v = l1 * uv1.v + l2 * uv2.v + l3 * uv3.v;
                                    //         let texture_width = texture.nd_img.shape()[0];
                                    //         let texture_height = texture.nd_img.shape()[1];
                                    //         Some(sample_nd_img(
                                    //             &texture.nd_img,
                                    //             u * texture_width as f64,
                                    //             v * texture_height as f64,
                                    //         ))
                                    //     }
                                    //     _ => None,
                                    // };

                                    // TODO: move into shader code
                                    // if let Some(albedo) = color_option {
                                    //     // dbg!(color);
                                    //     // let to_light_vec =
                                    //     //     Vector3::new(0.1, 0.1, time as f64 * 0.000001).normalize();
                                    //     let to_light_vec =
                                    //         Vector3::new(camera_pos.x, camera_pos.y, camera_pos.z)
                                    //             .normalize();
                                    //     // let diffuse_proportion = face_normal_vector.dot(&to_light_vec).max(0.0);
                                    //     // let diffuse_proportion = face_normal_vector.dot(&to_light_vec).abs();
                                    //     let diffuse_proportion =
                                    //         normal_vector.dot(&to_light_vec).max(0.0);

                                    //     let light_intensity = 1.0;
                                    //     let ambient_light = 1.0;
                                    //     let color = [
                                    //         (ambient_light
                                    //             + (diffuse_proportion
                                    //                 * light_intensity
                                    //                 * albedo[0]))
                                    //             .min(255.0),
                                    //         (ambient_light
                                    //             + (diffuse_proportion
                                    //                 * light_intensity
                                    //                 * albedo[1]))
                                    //             .min(255.0),
                                    //         (ambient_light
                                    //             + (diffuse_proportion
                                    //                 * light_intensity
                                    //                 * albedo[2]))
                                    //             .min(255.0),
                                    //         255.0,
                                    //     ];
                                    //     img.set(x, y, color);
                                    // }

                                    let texture_coordinate_interp =
                                        if let (Some(uv1), Some(uv2), Some(uv3)) = (
                                            vsr1.texture_coordinate,
                                            vsr2.texture_coordinate,
                                            vsr3.texture_coordinate,
                                        ) {
                                            Some(wavefront_obj::obj::TVertex {
                                                u: l1 * uv1.u + l2 * uv2.u + l3 * uv3.u,
                                                v: l1 * uv1.v + l2 * uv2.v + l3 * uv3.v,
                                                w: l1 * uv1.w + l2 * uv2.w + l3 * uv3.w,
                                            })
                                        } else {
                                            None
                                        };

                                    let color_interp = if let (Some(c1), Some(c2), Some(c3)) =
                                        (vsr1.color, vsr2.color, vsr3.color)
                                    {
                                        Some([
                                            l1 * c1[0] + l2 * c2[0] + l3 * c3[0],
                                            l1 * c1[1] + l2 * c2[1] + l3 * c3[1],
                                            l1 * c1[2] + l2 * c2[2] + l3 * c3[2],
                                            l1 * c1[3] + l2 * c2[3] + l3 * c3[3],
                                        ])
                                    } else {
                                        None
                                    };

                                    let fragment_shader_result =
                                        fragment_shader(FragmentShaderArgs {
                                            viewport_space_position,
                                            normal_interp,
                                            texture_coordinate_interp,
                                            color_interp,
                                            barycentric_coords,
                                        });
                                    if let Some(color) = fragment_shader_result.color {
                                        frame_buffer.set(x, y, color);
                                    }
                                }
                            },
                        );
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

            let long_side_segment = Segment3D::from((viewport_points[0], viewport_points[2]));
            let long_side_split_point = long_side_segment.get_point(
                (viewport_points[1].y - viewport_points[0].y)
                    / (viewport_points[2].y - viewport_points[0].y),
            )?;

            fill_half_triangle(
                Segment3D::from((viewport_points[0], viewport_points[1])),
                Segment3D::from((viewport_points[0], long_side_split_point)),
            )?;
            fill_half_triangle(
                Segment3D::from((viewport_points[2], viewport_points[1])),
                Segment3D::from((viewport_points[2], long_side_split_point)),
            )?;

            Ok(())
        })
        .find_map(|result| -> Option<anyhow::Result<()>> {
            match result {
                Ok(_) => None,
                Err(err) => Some(Err(err)),
            }
        });
}

// impl<'a> ModelRenderer {
//     pub fn render<F: FnMut(wavefront_obj::obj::Vertex) -> VertexShaderResult>(
//         &mut self,
//         vertex_shader: F,
//         fragment_shader: fn(wavefront_obj::obj::Vertex) -> FragmentShaderResult,
//     ) {
//         let mut vertex_shader_results = self.model.geometry[0].shapes.iter().map(vertex_shader);
//     }
// }

// struct VertexShader<T>(fn(wavefront_obj::obj::Vertex) -> VertexShaderResult<T>);

pub struct VertexShaderArgs {
    pub model_position: nalgebra::Vector3<f64>,
    pub normal: nalgebra::Vector3<f64>,
    pub texture_coordinate: Option<wavefront_obj::obj::TVertex>,
}

impl
    From<(
        wavefront_obj::obj::Vertex,
        nalgebra::Vector3<f64>,
        Option<wavefront_obj::obj::TVertex>,
    )> for VertexShaderArgs
{
    fn from(
        (position, normal, texture_coordinate): (
            wavefront_obj::obj::Vertex,
            nalgebra::Vector3<f64>,
            Option<wavefront_obj::obj::TVertex>,
        ),
    ) -> VertexShaderArgs {
        VertexShaderArgs {
            model_position: nalgebra::Vector3::new(position.x, position.y, position.z),
            normal,
            texture_coordinate,
        }
    }
}

pub struct VertexShaderResult {
    // vertex: wavefront_obj::obj::Vertex,
    pub clip_space_position: nalgebra::Vector3<f64>,
    pub world_space_position: nalgebra::Vector3<f64>,

    pub normal: nalgebra::Vector3<f64>,
    pub t_vector: Option<nalgebra::Vector3<f64>>,
    pub bt_vector: Option<nalgebra::Vector3<f64>>,

    pub texture_coordinate: Option<wavefront_obj::obj::TVertex>,

    pub color: Option<[f64; 4]>,
}

pub struct FragmentShaderArgs {
    // pub vertex_shader_result: VertexShaderResult,
    pub viewport_space_position: nalgebra::Vector3<f64>,

    pub normal_interp: nalgebra::Vector3<f64>,
    pub texture_coordinate_interp: Option<wavefront_obj::obj::TVertex>,
    pub color_interp: Option<[f64; 4]>,
    pub barycentric_coords: (f64, f64, f64),
}

// struct FragmentShader(fn(wavefront_obj::obj::Vertex) -> FragmentShaderResult);

pub struct FragmentShaderResult {
    pub color: Option<[f64; 4]>,
}
