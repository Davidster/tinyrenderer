use super::helpers::*;

use std::marker::PhantomData;

pub struct ModelRendererState {
    pub frame_buffer: MyRgbaImage,
    // pub z_buffer: MyGrayImage,
    pub texture: Option<MyRgbaImage>,
    pub normal_map: Option<MyRgbaImage>,
    pub model: wavefront_obj::obj::Object,
    // phantom: PhantomData<'a>,
}

pub fn render_model<
    // Option<nalgebra::Vector3<f64>, Option<wavefront_obj::obj::TVertex>
    V: FnMut(VertexShaderArgs) -> VertexShaderResult,
    F: FnMut(wavefront_obj::obj::Vertex, VertexShaderResult) -> FragmentShaderResult,
>(
    model_renderer_state: &mut ModelRendererState,
    vertex_shader: &mut V,
    fragment_shader: &mut F,
) {
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
                    let mut get_normal_for_vertex =
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
    let fragment_shader_results = vertex_shader_results
        .iter()
        .flatten()
        .map(|vertex_shader_results| {
            // TODO: rasterize then call fragment shader on each pixel
        })
        .collect();
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
    clip_space_position: nalgebra::Vector3<f64>,
    world_space_position: nalgebra::Vector3<f64>,
    normal: nalgebra::Vector3<f64>,
    color: Option<[f64; 4]>,
    t_vector: Option<nalgebra::Vector3<f64>>,
    bt_vector: Option<nalgebra::Vector3<f64>>,
}

pub struct FragmentShaderArgs {
    pub vertex_shader_result: VertexShaderResult,
}

// struct FragmentShader(fn(wavefront_obj::obj::Vertex) -> FragmentShaderResult);

pub struct FragmentShaderResult {
    color: [u8; 4],
}
