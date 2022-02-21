use std::cell::Cell;

use super::*;

type Vector3 = nalgebra::Vector3<f64>;
type Matrix4 = nalgebra::Matrix4<f64>;

#[derive(Clone, Debug)]
pub struct Transform {
    pub position: Cell<Vector3>,
    pub rotation: Cell<Vector3>, // euler angles
    pub scale: Cell<Vector3>,
    pub matrix: Cell<Matrix4>,
}

impl Transform {
    pub fn new() -> Transform {
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
