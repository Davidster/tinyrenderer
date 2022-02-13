mod helpers;
mod model_renderer;
mod segment_3d;

use std::f64::consts::PI;

use helpers::*;
use model_renderer::*;

#[show_image::main]
fn main() {
    let mut african_head_mesh_component = MeshComponent::from((
        "./src/african_head/african_head.obj",
        "./src/african_head/african_head_diffuse.png",
        "./src/african_head/african_head_nm_tangent.png",
    ));

    let mut african_head_eye_inner_mesh_component = MeshComponent::from((
        "./src/african_head/african_head_eye_inner.obj",
        "./src/african_head/african_head_eye_inner_diffuse.png",
        "./src/african_head/african_head_eye_inner_nm_tangent.png",
    ));

    let mut african_head_eye_outer_mesh_component = MeshComponent::from((
        "./src/african_head/african_head_eye_outer.obj",
        "./src/african_head/african_head_eye_outer_diffuse.png",
        "./src/african_head/african_head_eye_outer_nm_tangent.png",
    ));

    let mut rando_mesh_component = get_rando_mesh_component();

    african_head_eye_inner_mesh_component.parent = Some(&african_head_mesh_component);
    african_head_eye_outer_mesh_component.parent = Some(&african_head_mesh_component);

    let frame_width = 1000;
    let frame_height = 1000;
    let mut model_renderer_state = ModelRendererState::new(frame_width, frame_height);

    clear_screen(&mut model_renderer_state);

    let window = show_image::create_window(
        "img",
        show_image::WindowOptions::new().set_size([frame_width as u32, frame_height as u32]),
    )
    .expect("Failed to create window");

    let mut triangle_index = 0;
    let mut time = 0;
    let mut showing_normal_map = false;
    let colors = [RED, BLUE, GREEN];

    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();

    loop {
        let before = std::time::Instant::now();
        time += 1;
        if time % 10 == 0 {
            showing_normal_map = !showing_normal_map;
        }
        clear_screen(&mut model_renderer_state);

        let camera_direction = nalgebra::Vector3::new(0.0, 0.0, 1.0).normalize();
        let camera_direction_scaled = camera_direction * 2.0;
        let camera_pos = nalgebra::Point3::from(camera_direction_scaled);

        let camera_direction_matrix = nalgebra::Matrix4::look_at_rh(
            &camera_pos,
            &nalgebra::Point3::new(0.0, 0.0, 0.0),
            &nalgebra::Vector3::new(0.0, 1.0, 0.0),
        );

        african_head_mesh_component
            .transform
            .set_rotation(nalgebra::Vector3::new(0.025 * (time as f64), 0.0, 0.0));
        // let rotation_matrix = make_rotation_matrix(0.025 * (time as f64), 0.0, 0.0);
        // let rotation_matrix = make_rotation_matrix(0.0, 0.0, 0.0);
        // african_head_mesh_component
        //     .transform
        //     .set_position(nalgebra::Vector3::new(0.025 * (time as f64), 0.0, 0.0));

        // let scale = 1.0 + 0.0025 * (time as f64);
        // african_head_mesh_component
        //     .transform
        //     .set_scale(nalgebra::Vector3::new(scale, scale, scale));

        // african_head_eye_inner_mesh_component
        //     .transform
        //     .set_position(nalgebra::Vector3::new(0.0, 0.0, 0.0001 * (time as f64)));
        // african_head_eye_inner_mesh_component
        //     .transform
        //     .set_scale(nalgebra::Vector3::new(scale, scale, scale));
        // let translation_matrix = make_translation_matrix(nalgebra::Vector3::new(0.0, 0.0, 0.0));
        // let translation_matrix =
        //     make_translation_matrix(camera_direction * (-0.004 * time as f64));

        // let model_view_matrix = camera_direction_matrix * translation_matrix * rotation_matrix;
        let horizontal_fov = 0.5 * PI;
        let perspective_matrix = make_perspective_matrix(
            10.0,
            1.0,
            horizontal_fov,
            horizontal_fov * (frame_height as f64 / frame_width as f64),
        );

        let perspective_matrix_clone = perspective_matrix.clone();

        let mut do_render_mesh_component = |mesh_component: &MeshComponent| {
            // let model_matrix_1 = mesh_component.local_to_world_matrix();
            // let rotation = mesh_component.transform.rotation.get();
            // let model_matrix_2 = make_translation_matrix(mesh_component.transform.position.get())
            //     * make_rotation_matrix(rotation.x, rotation.y, rotation.z)
            //     * make_scale_matrix(mesh_component.transform.scale.get());
            // dbg!(model_matrix_1);
            // dbg!(model_matrix_2);
            let model_view_matrix =
                camera_direction_matrix * mesh_component.local_to_world_matrix();

            let inverse_model_view_matrix = model_view_matrix
                .transpose()
                .try_inverse()
                .unwrap_or_else(|| {
                    println!("Failed to transform normal for: {:?}", model_view_matrix);
                    nalgebra::Matrix4::identity()
                });

            let model_view_matrix_clone = model_view_matrix.clone();
            let inverse_model_view_matrix_clone = inverse_model_view_matrix.clone();

            let normal_map = mesh_component.normal_map.clone();
            let texture = mesh_component.texture.clone();

            render_mesh_component(
                &mut model_renderer_state,
                &mesh_component.mesh,
                Box::new(
                    move |VertexShaderArgs {
                              local_position,
                              local_normal,
                              texture_coordinate,
                              ..
                          }| {
                        let global_position =
                            perspective_matrix_clone * model_view_matrix_clone * local_position;
                        let global_position_x = global_position.x / global_position.w;
                        let global_position_y = global_position.y / global_position.w;
                        let global_position_z = global_position.z / global_position.w;
                        let clip_space_position = nalgebra::Vector4::new(
                            global_position_x,
                            global_position_y,
                            global_position_z,
                            1.0,
                        );

                        let normal = inverse_model_view_matrix_clone * local_normal;

                        VertexShaderResult {
                            clip_space_position,
                            normal,
                            texture_coordinate,
                            color: Some(WHITE),
                        }
                    },
                ),
                Box::new(
                    move |FragmentShaderArgs {
                              //   viewport_space_position,
                              t_bt_vectors,
                              normal_interp,
                              texture_coordinate_interp,
                              color_interp,
                              //   barycentric_coords,
                              ..
                          }| {
                        let normal_vector = match (texture_coordinate_interp, t_bt_vectors) {
                            (
                                Some(wavefront_obj::obj::TVertex { u, v, .. }),
                                Some((t_vector, bt_vector)),
                            ) => {
                                // dbg!(u, v, normal_map.nd_img.shape());
                                let normal_map_width = normal_map.nd_img.shape()[0];
                                let normal_map_height = normal_map.nd_img.shape()[1];
                                let normal_map_normal_rgb = sample_nd_img(
                                    &normal_map.nd_img,
                                    u * (normal_map_width - 1) as f64,
                                    v * (normal_map_height - 1) as f64,
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
                                    normal_interp.x,
                                    t_vector.y,
                                    bt_vector.y,
                                    normal_interp.y,
                                    t_vector.z,
                                    bt_vector.z,
                                    normal_interp.z,
                                );
                                (tbn_mat * normal_map_normal).normalize()
                            }
                            _ => normal_interp,
                        };

                        // let albedo_color_option = Some(WHITE);
                        let albedo_color_option =
                            if let Some(wavefront_obj::obj::TVertex { u, v, .. }) =
                                texture_coordinate_interp
                            {
                                let texture_width = texture.nd_img.shape()[0];
                                let texture_height = texture.nd_img.shape()[1];
                                // if u > 0.99 || v > 0.99 {
                                //     dbg!(
                                //         u,
                                //         v,
                                //         u * (texture_width - 1) as f64,
                                //         v * (texture_height - 1) as f64,
                                //         model.texture.nd_img.shape()
                                //     );
                                //     std::thread::sleep(std::time::Duration::from_millis(500));
                                // }

                                Some(sample_nd_img(
                                    &texture.nd_img,
                                    u * (texture_width - 1) as f64,
                                    v * (texture_height - 1) as f64,
                                ))
                            } else if let Some(color) = color_interp {
                                Some(color)
                            } else {
                                None
                            };
                        let color = albedo_color_option.map(|albedo| {
                            let to_light_vec =
                                nalgebra::Vector3::new(camera_pos.x, camera_pos.y, camera_pos.z)
                                    .normalize();
                            let diffuse_proportion = normal_vector.dot(&to_light_vec).max(0.0);

                            let light_intensity = 1.0;
                            let ambient_light = 1.0;
                            [
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
                            ]
                        });
                        FragmentShaderResult { color }
                    },
                ),
            );
        };

        // do_render_mesh_component(&rando_mesh_component);
        do_render_mesh_component(&african_head_mesh_component);
        do_render_mesh_component(&african_head_eye_inner_mesh_component);
        // do_render_mesh_component(&african_head_eye_outer_mesh_component);

        dbg!(time);
        dbg!(before.elapsed());
        if time == 250 {
            break;
        }
        window
            .set_image(
                "img",
                ndarray_to_image_rgba(&flip_vertically(&model_renderer_state.frame_buffer.nd_img)),
            )
            .expect("Failed to set image");

        // window
        //     .set_image(
        //         "img",
        //         ndarray_to_image_gray(
        //             &flip_vertically(&get_drawable_z_buffer(
        //                 &model_renderer_state.z_buffer.nd_img,
        //             )),
        //             ImgConversionType::NORMALIZE,
        //         ),
        //     )
        //     .expect("Failed to set image");
        // thread::sleep(Duration::from_millis(50));
    }
}
