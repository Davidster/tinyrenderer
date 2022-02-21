use super::*;

type Matrix4 = nalgebra::Matrix4<f64>;

#[derive(Clone, Debug)]
pub struct MeshComponent<'a> {
    pub mesh: wavefront_obj::obj::Object,
    pub transform: Transform,
    pub texture: MyRgbaImage,
    pub normal_map: MyRgbaImage,
    pub parent: Option<&'a MeshComponent<'a>>,
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
