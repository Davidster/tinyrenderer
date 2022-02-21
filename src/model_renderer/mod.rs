use super::helpers::*;
use super::segment_3d::*;

use std::cell::Cell;

use anyhow::Result;
use nalgebra::matrix;

type Vector3 = nalgebra::Vector3<f64>;
type Matrix4 = nalgebra::Matrix4<f64>;

#[derive(Clone, Debug)]
pub struct MeshComponent<'a> {
    pub mesh: wavefront_obj::obj::Object,
    pub transform: Transform,
    pub texture: MyRgbaImage,
    pub normal_map: MyRgbaImage,
    pub parent: Option<&'a MeshComponent<'a>>,
}

#[derive(Clone, Debug)]
pub struct Transform {
    pub position: Cell<Vector3>,
    pub rotation: Cell<Vector3>, // euler angles
    pub scale: Cell<Vector3>,
    pub matrix: Cell<Matrix4>,
}

#[derive(Clone, Debug)]
pub struct ModelRendererState {
    pub frame_buffer: MyRgbaImage,
    pub z_buffer: MyGrayImage,
    clear_frame_buffer: MyRgbaImage,
    clear_z_buffer: MyGrayImage,
}

impl<'a, P: AsRef<std::path::Path>> From<(P, P, P)> for MeshComponent<'a> {
    fn from((obj_path, texture_path, normal_map_path): (P, P, P)) -> MeshComponent<'a> {
        let obj = load_model(obj_path).unwrap();
        let texture = flip_vertically(&load_nd_rgba_img_from_file(texture_path).unwrap());
        let normal_map = flip_vertically(&load_nd_rgba_img_from_file(normal_map_path).unwrap());
        MeshComponent {
            mesh: obj,
            transform: Transform::new(),
            texture: MyRgbaImage { nd_img: texture },
            normal_map: MyRgbaImage { nd_img: normal_map },
            parent: None,
        }
    }
}

impl<'a> From<(wavefront_obj::obj::Object, NDRgbaImage, NDRgbaImage)> for MeshComponent<'a> {
    fn from(
        (obj, texture, normal_map): (wavefront_obj::obj::Object, NDRgbaImage, NDRgbaImage),
    ) -> MeshComponent<'a> {
        MeshComponent {
            mesh: obj,
            transform: Transform::new(),
            texture: MyRgbaImage { nd_img: texture },
            normal_map: MyRgbaImage { nd_img: normal_map },
            parent: None,
        }
    }
}

impl<'a> MeshComponent<'a> {
    pub fn local_to_world_matrix(&self) -> Matrix4 {
        let mut transforms: Vec<&Transform> = Vec::new();
        let mut current_component: &MeshComponent = self;
        loop {
            transforms.push(&current_component.transform);
            match current_component.parent {
                Some(component) => {
                    current_component = component;
                }
                None => {
                    break;
                }
            }
        }
        transforms
            .iter()
            .fold(Matrix4::identity(), |acc, curr| acc * curr.matrix.get())
    }
}

impl Transform {
    fn new() -> Transform {
        Transform {
            position: Cell::new(Vector3::new(0.0, 0.0, 0.0)),
            rotation: Cell::new(Vector3::new(0.0, 0.0, 0.0)),
            scale: Cell::new(Vector3::new(1.0, 1.0, 1.0)),
            matrix: Cell::new(Matrix4::identity()),
        }
    }

    #[allow(dead_code)]
    pub fn position(&self) -> Vector3 {
        self.position.get()
    }

    #[allow(dead_code)]
    pub fn rotation(&self) -> Vector3 {
        self.rotation.get()
    }

    #[allow(dead_code)]
    pub fn scale(&self) -> Vector3 {
        self.scale.get()
    }

    #[allow(dead_code)]
    pub fn matrix(&self) -> Matrix4 {
        self.matrix.get()
    }

    #[allow(dead_code)]
    pub fn set_position(&self, new_position: Vector3) {
        self.position.set(new_position);
        let mut matrix = self.matrix.get();
        matrix.m14 = new_position.x;
        matrix.m24 = new_position.y;
        matrix.m34 = new_position.z;
        self.matrix.set(matrix);
    }

    #[allow(dead_code)]
    pub fn set_rotation(&self, new_rotation: Vector3) {
        self.rotation.set(new_rotation);
        self.resync_matrix();
    }

    #[allow(dead_code)]
    pub fn set_scale(&self, new_scale: Vector3) {
        self.scale.set(new_scale);
        self.resync_matrix();
    }

    #[allow(dead_code)]
    fn resync_matrix(&self) {
        let rotation = self.rotation.get();
        self.matrix.set(
            make_translation_matrix(self.position.get())
                * make_rotation_matrix(rotation.x, rotation.y, rotation.z)
                * make_scale_matrix(self.scale.get()),
        );
    }
}

impl ModelRendererState {
    pub fn new(frame_width: usize, frame_height: usize) -> ModelRendererState {
        let mut frame_buffer = MyRgbaImage {
            nd_img: ndarray::Array3::zeros((frame_width, frame_height, 4)),
        };
        let mut z_buffer = MyGrayImage {
            nd_img: ndarray::Array3::zeros((frame_width, frame_height, 1)),
        };
        for x in 0..frame_width {
            for y in 0..frame_height {
                // set all to black
                frame_buffer.set(x, y, BLACK);
                // reset z-buffer
                z_buffer.set(x, y, [f64::NEG_INFINITY]);
            }
        }
        ModelRendererState {
            clear_frame_buffer: frame_buffer.clone(),
            clear_z_buffer: z_buffer.clone(),
            frame_buffer,
            z_buffer,
        }
    }
}

pub fn _draw_line(
    img: &mut MyRgbaImage,
    start_point: Vector3,
    end_point: Vector3,
    color: [f64; 4],
) -> Result<()> {
    let mut points = [start_point, end_point].to_vec();
    points.sort_by(|a, b| b.x.partial_cmp(&a.x).unwrap_or(std::cmp::Ordering::Equal));
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

pub fn _clear_screen_old_and_slow(model_renderer_state: &mut ModelRendererState) {
    let frame_buffer = &mut model_renderer_state.frame_buffer;
    let z_buffer = &mut model_renderer_state.z_buffer;
    let frame_width = frame_buffer.nd_img.shape()[0];
    let frame_height = frame_buffer.nd_img.shape()[1];
    for x in 0..frame_width {
        for y in 0..frame_height {
            // set all to black
            frame_buffer.set(x, y, BLACK);
            // reset z-buffer
            z_buffer.set(x, y, [f64::NEG_INFINITY]);
        }
    }
}

pub fn clear_screen(model_renderer_state: &mut ModelRendererState) {
    model_renderer_state
        .frame_buffer
        .nd_img
        .clone_from(&model_renderer_state.clear_frame_buffer.nd_img);
    model_renderer_state
        .z_buffer
        .nd_img
        .clone_from(&model_renderer_state.clear_z_buffer.nd_img);
}

// TODO: draw bounding box around the triangle and render in horizontal or vertical lines.
//       this may help improve cache coherency
//       ...
//       After all, I'm  not D
pub fn render_mesh_component(
    model_renderer_state: &mut ModelRendererState,
    mesh: &wavefront_obj::obj::Object,
    vertex_shader: &mut dyn Fn(VertexShaderArgs) -> VertexShaderResult,
    fragment_shader: &mut dyn Fn(FragmentShaderArgs) -> FragmentShaderResult,
) {
    let frame_width = model_renderer_state.frame_buffer.nd_img.shape()[0];
    let frame_height = model_renderer_state.frame_buffer.nd_img.shape()[1];
    // let texture_option = &model_renderer_state.texture;
    // let normal_map_option = &model_renderer_state.normal_map;
    let frame_buffer = &mut model_renderer_state.frame_buffer;
    let z_buffer = &mut model_renderer_state.z_buffer;
    let shapes = &mesh.geometry[0].shapes;
    let run_vertex_shader = |shape: &wavefront_obj::obj::Shape| {
        match shape.primitive {
            wavefront_obj::obj::Primitive::Triangle(i1, i2, i3) => {
                // get triangle vertices and run them through the vertex shader
                let v1 = mesh.vertices[i1.0];
                let v2 = mesh.vertices[i2.0];
                let v3 = mesh.vertices[i3.0];
                let get_face_normal = || {
                    let side_1 =
                        nalgebra::Vector3::new(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z).normalize();
                    let side_2 =
                        nalgebra::Vector3::new(v3.x - v2.x, v3.y - v2.y, v3.z - v2.z).normalize();
                    let face_normal = side_1.cross(&side_2).normalize();
                    nalgebra::Vector4::new(face_normal.x, face_normal.y, face_normal.z, 0.0)
                };
                let get_normal_for_vertex = |(_, _, index_option): wavefront_obj::obj::VTNIndex| {
                    index_option
                        .map(|index| mesh.normals[index])
                        .map(|normal| nalgebra::Vector4::new(normal.x, normal.y, normal.z, 0.0))
                        .unwrap_or_else(get_face_normal)
                };

                let n1 = get_normal_for_vertex(i1);
                let n2 = get_normal_for_vertex(i2);
                let n3 = get_normal_for_vertex(i3);

                let get_texture_for_vertex =
                    |(_, index_option, _): wavefront_obj::obj::VTNIndex| {
                        index_option.map(|index| mesh.tex_vertices[index])
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
    };

    let vertex_shader_results: Vec<
        Option<(VertexShaderResult, VertexShaderResult, VertexShaderResult)>,
    > = shapes.iter().map(run_vertex_shader).collect();
    // let before = std::time::Instant::now();

    let run_fragment_shader = |(vsr1, vsr2, vsr3): &(
        VertexShaderResult,
        VertexShaderResult,
        VertexShaderResult,
    )| {
        // dbg!((vsr1, vsr2, vsr3));
        let should_cull_point = |point: &nalgebra::Vector4<f64>| {
            point.x < -1.0
                || point.x > 1.0
                || point.y < -1.0
                || point.y > 1.0
                || point.z < -1.0
                || point.z > 1.0
        };

        let points = [
            vsr1.clip_space_position,
            vsr2.clip_space_position,
            vsr3.clip_space_position,
        ]
        .to_vec();

        if points.iter().all(|point| should_cull_point(point)) {
            // return Ok(());
            return;
        }

        let viewport_matrix = make_scale_matrix(nalgebra::Vector3::new(
            (frame_width - 1) as f64 / 2.0,
            (frame_height - 1) as f64 / 2.0,
            1.0,
        )) * make_translation_matrix(nalgebra::Vector3::new(1.0, 1.0, 0.0));

        let mut viewport_points: Vec<nalgebra::Vector4<f64>> = points
            .iter()
            .map(|point| viewport_matrix * nalgebra::Vector4::new(point.x, point.y, point.z, 1.0))
            // TODO: is this needed?
            .map(|point| {
                nalgebra::Vector4::new(point.x / point.w, point.y / point.w, point.z / point.w, 1.0)
            })
            .collect();

        let viewport_points_unsorted = viewport_points.clone();

        viewport_points.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal));

        let t_bt_vectors = match (
            vsr1.texture_coordinate,
            vsr2.texture_coordinate,
            vsr3.texture_coordinate,
        ) {
            (Some(uv1), Some(uv2), Some(uv3)) => {
                let uv_mat = matrix![uv2.u - uv1.u, uv2.v - uv1.v;
                                    uv3.u - uv1.u, uv3.v - uv1.v;];
                let triangle_corner_edges = matrix![points[1].x - points[0].x, points[1].y - points[0].y, points[1].z - points[0].z;
                                                    points[2].x - points[0].x, points[2].y - points[0].y, points[2].z - points[0].z;];
                match uv_mat.try_inverse() {
                    Some(uv_mat_inv) => {
                        let tb_mat = uv_mat_inv * triangle_corner_edges;
                        let t_vector =
                            nalgebra::Vector4::new(tb_mat.m11, tb_mat.m12, tb_mat.m13, 0.0)
                                .normalize();
                        let bt_vector =
                            nalgebra::Vector4::new(tb_mat.m21, tb_mat.m22, tb_mat.m23, 0.0)
                                .normalize();
                        Some((t_vector, bt_vector))
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
            let error_option = segment_a
                .get_point_iterator(resolution)?
                .zip(segment_b.get_point_iterator(resolution)?)
                .map(|(segment_a_point, segment_b_point)| -> anyhow::Result<()> {
                    let horizontal_segment = Segment3D::new(segment_a_point, segment_b_point);
                    horizontal_segment.get_point_iterator(None)?.for_each(
                        |viewport_space_position| {
                            let x = viewport_space_position.x.round() as i64;
                            let y = viewport_space_position.y.round() as i64;
                            let dist = -viewport_space_position.z;
                            if x < 0
                                || x > frame_width as i64 - 1
                                || y < 0
                                || y > frame_height as i64 - 1
                            {
                                return;
                            }

                            let current_z = z_buffer.get(x as usize, y as usize)[0];
                            if current_z == f64::NEG_INFINITY || current_z > dist {
                                z_buffer.set(x as usize, y as usize, [dist]);
                                let (
                                    barycentric_coords,
                                    normal_interp,
                                    texture_coordinate_interp,
                                    color_interp,
                                ) = get_fragment_shader_elements(
                                    &viewport_points_unsorted,
                                    viewport_space_position,
                                    vsr1,
                                    vsr2,
                                    vsr3,
                                );

                                // let (barycentric_coords, normal_interp, texture_coordinate_interp, color_interp) = get_fragment_shader_elements();

                                let fragment_shader_result = fragment_shader(FragmentShaderArgs {
                                    viewport_space_position,
                                    t_bt_vectors,
                                    normal_interp,
                                    texture_coordinate_interp,
                                    color_interp,
                                    barycentric_coords,
                                });
                                if let Some(color) = fragment_shader_result.color {
                                    frame_buffer.set(x as usize, y as usize, color);
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
        let long_side_split_point = long_side_segment
            .get_point(
                (viewport_points[1].y - viewport_points[0].y)
                    / (viewport_points[2].y - viewport_points[0].y),
            )
            .unwrap();

        fill_half_triangle(
            Segment3D::from((viewport_points[0], viewport_points[1])),
            Segment3D::from((viewport_points[0], long_side_split_point)),
        )
        .unwrap();
        fill_half_triangle(
            Segment3D::from((viewport_points[2], viewport_points[1])),
            Segment3D::from((viewport_points[2], long_side_split_point)),
        )
        .unwrap();

        // Ok(())
    };

    // let error_option = vertex_shader_results.iter()...
    vertex_shader_results
        .iter()
        .flatten()
        .for_each(run_fragment_shader);

    // dbg!(before.elapsed());
    // .find_map(|result| -> Option<anyhow::Result<()>> {
    //     match result {
    //         Ok(_) => None,
    //         Err(err) => Some(Err(err)),
    //     }
    // });
    // if let Some(err) = error_option {
    //     err.unwrap();
    // }

    // mesh_components.iter().for_each(|mesh_component| run_shaders_on_mesh_component(mesh_component));
}

fn get_fragment_shader_elements(
    viewport_points_unsorted: &Vec<nalgebra::Vector4<f64>>,
    viewport_space_position: nalgebra::Vector3<f64>,
    vsr1: &VertexShaderResult,
    vsr2: &VertexShaderResult,
    vsr3: &VertexShaderResult,
) -> (
    (f64, f64, f64),
    nalgebra::Vector3<f64>,
    Option<wavefront_obj::obj::TVertex>,
    Option<[f64; 4]>,
) {
    let barycentric_coords = get_barycentric_coords_for_point_in_triangle(
        (
            viewport_points_unsorted[0],
            viewport_points_unsorted[1],
            viewport_points_unsorted[2],
        ),
        viewport_space_position,
    );
    let (l1, l2, l3) = barycentric_coords;
    let (n1, n2, n3) = (vsr1.normal, vsr2.normal, vsr3.normal);
    let normal_interp = nalgebra::Vector3::new(
        l1 * n1[0] + l2 * n2[0] + l3 * n3[0],
        l1 * n1[1] + l2 * n2[1] + l3 * n3[1],
        l1 * n1[2] + l2 * n2[2] + l3 * n3[2],
    )
    .normalize();
    let texture_coordinate_interp = if let (Some(uv1), Some(uv2), Some(uv3)) = (
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
    let color_interp = if let (Some(c1), Some(c2), Some(c3)) = (vsr1.color, vsr2.color, vsr3.color)
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
    (
        barycentric_coords,
        normal_interp,
        texture_coordinate_interp,
        color_interp,
    )
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

#[derive(Clone, Debug)]
pub struct VertexShaderArgs {
    pub local_position: nalgebra::Vector4<f64>,
    pub local_normal: nalgebra::Vector4<f64>,
    pub texture_coordinate: Option<wavefront_obj::obj::TVertex>,
}

#[derive(Clone, Debug)]
pub struct VertexShaderResult {
    // vertex: wavefront_obj::obj::Vertex,
    pub clip_space_position: nalgebra::Vector4<f64>,
    // pub world_space_position: Vector3,
    pub normal: nalgebra::Vector4<f64>,
    pub texture_coordinate: Option<wavefront_obj::obj::TVertex>,
    pub color: Option<[f64; 4]>,
}

#[derive(Clone, Debug)]
pub struct FragmentShaderArgs {
    // pub vertex_shader_result: VertexShaderResult,
    pub viewport_space_position: Vector3,
    pub t_bt_vectors: Option<(nalgebra::Vector4<f64>, nalgebra::Vector4<f64>)>,
    pub normal_interp: Vector3,
    pub texture_coordinate_interp: Option<wavefront_obj::obj::TVertex>,
    pub color_interp: Option<[f64; 4]>,
    pub barycentric_coords: (f64, f64, f64),
}

// struct FragmentShader(fn(wavefront_obj::obj::Vertex) -> FragmentShaderResult);

#[derive(Clone, Debug)]
pub struct FragmentShaderResult {
    pub color: Option<[f64; 4]>,
}

impl
    From<(
        wavefront_obj::obj::Vertex,
        nalgebra::Vector4<f64>,
        Option<wavefront_obj::obj::TVertex>,
    )> for VertexShaderArgs
{
    fn from(
        (position, model_normal, texture_coordinate): (
            wavefront_obj::obj::Vertex,
            nalgebra::Vector4<f64>,
            Option<wavefront_obj::obj::TVertex>,
        ),
    ) -> VertexShaderArgs {
        VertexShaderArgs {
            local_position: nalgebra::Vector4::new(position.x, position.y, position.z, 1.0),
            local_normal: model_normal,
            texture_coordinate,
        }
    }
}
