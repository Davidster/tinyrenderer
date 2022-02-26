mod helpers;
mod mesh_component;
mod texture;
mod transform;

use helpers::*;
use mesh_component::*;
use transform::*;

use image::GenericImageView;
use wgpu::util::DeviceExt;

const FRAME_WIDTH: i64 = 1080;
const FRAME_HEIGHT: i64 = 1080;

struct ShapeBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    num_vertices: u32,
}

enum ChosenShape {
    PENTAGON,
    STAR,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    current_mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,
    chosen_shape: ChosenShape,

    star_render_pipeline: wgpu::RenderPipeline,
    star_shape_buffers: ShapeBuffers,
    star_diffuse_texture_bind_group: wgpu::BindGroup,
    star_diffuse_texture: texture::Texture,

    pentagon_render_pipeline: wgpu::RenderPipeline,
    pentagon_shape_buffers: ShapeBuffers,
    pentagon_diffuse_texture_bind_group: wgpu::BindGroup,
    pentagon_diffuse_texture: texture::Texture,
}

type VertexPosition = [f32; 3];
type VertexColor = [f32; 3];
type VertexTextureCoords = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ColoredVertex {
    position: VertexPosition,
    color: VertexColor,
}

unsafe impl bytemuck::Pod for ColoredVertex {}
unsafe impl bytemuck::Zeroable for ColoredVertex {}

impl ColoredVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ColoredVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct TexturedVertex {
    position: VertexPosition,
    tex_coords: VertexTextureCoords,
}

unsafe impl bytemuck::Pod for TexturedVertex {}
unsafe impl bytemuck::Zeroable for TexturedVertex {}

impl TexturedVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TexturedVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

fn to_srgb(val: f32) -> f32 {
    val.powf(2.2)
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        let star_color = [
            to_srgb(0.9686274509803922),
            to_srgb(0.8745098039215686),
            to_srgb(0.11764705882352941),
        ];
        // make a perfect start of David
        // 1 / 6*tan(30deg)
        let t_2_offset = 1.0 / (6.0 * ((std::f32::consts::PI * 2.0 * 30.0) / 360.0));
        // a^2 + b^2 = c^2 -> b = sqrt(c^2 - a^2) where a is 0.5 and c is 1
        let t_height = ((1.0 - (0.5 * 0.5)) as f32).sqrt();
        let half_t_height = t_height / 2.0;
        let star_height = t_2_offset + t_height;

        // triangle pointing upwards
        let t_1 = vec![
            [-0.5, -half_t_height],
            [0.5, -half_t_height],
            [0.0, half_t_height], // upper tip
        ];
        // triangle pointing downwards
        let t_2 = vec![
            [-0.5, half_t_height],
            [0.5, half_t_height],
            [0.0, -half_t_height], // lower tip
        ];

        let t_2_translated: Vec<[f32; 2]> = t_2
            .iter()
            .map(|val| [val[0], val[1] - t_2_offset])
            .collect();
        let star_vertices: Vec<TexturedVertex> = t_1
            .iter()
            .chain(t_2_translated.iter())
            .map(|pos| {
                // move it up so it's centered at the origin
                [pos[0], pos[1] + (t_2_offset / 2.0), 0.0]
            })
            .map(|position| TexturedVertex {
                position,
                tex_coords: [
                    (position[0] + 0.5),
                    (position[1] + (star_height / 2.0)) / star_height,
                ],
                // tex_coords: pos
                //     .to_vec()
                //     .iter()
                //     .map(|val| (val + 1.0) / 2.0)
                //     .collect::<Vec<f32>>()
                //     .as_slice()
                //     .try_into()
                //     .unwrap(),
            })
            .collect();

        dbg!(&star_vertices);

        let star_indices: &[u16] = &[0, 1, 2, 3, 5, 4];

        let pentagon_vertices: Vec<TexturedVertex> = vec![
            TexturedVertex {
                position: [-0.0868241, 0.49240386, 0.0],
                tex_coords: [0.4131759, 0.99240386],
            },
            TexturedVertex {
                position: [-0.49513406, 0.06958647, 0.0],
                tex_coords: [0.0048659444, 0.56958647],
            },
            TexturedVertex {
                position: [-0.21918549, -0.44939706, 0.0],
                tex_coords: [0.28081453, 0.05060294],
            },
            TexturedVertex {
                position: [0.35966998, -0.3473291, 0.0],
                tex_coords: [0.85967, 0.1526709],
            },
            TexturedVertex {
                position: [0.44147372, 0.2347359, 0.0],
                tex_coords: [0.9414737, 0.7347359],
            },
        ]
        .iter()
        .map(
            |TexturedVertex {
                 position,
                 tex_coords,
             }| TexturedVertex {
                position: *position,
                tex_coords: [tex_coords[0], 1.0 - tex_coords[1]],
            },
        )
        .collect();

        let pentagon_indices: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

        let backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(backends);
        let size = window.inner_size();
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");
        let adapter_info = adapter.get_info();
        println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let swapchain_format = surface.get_preferred_format(&adapter).unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &config);

        let star_texture_bytes = include_bytes!("starofdavid2.png");
        let star_texture =
            texture::Texture::from_bytes(&device, &queue, star_texture_bytes, "starofdavid.png")
                .unwrap();

        let star_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("star texture_bind_group_layout"),
            });

        let star_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &star_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&star_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&star_texture.sampler),
                },
            ],
            label: Some("star diffuse_bind_group"),
        });

        let tree_texture_bytes = include_bytes!("happy-tree.png");
        let tree_texture =
            texture::Texture::from_bytes(&device, &queue, tree_texture_bytes, "happy-tree.png")
                .unwrap();

        let tree_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("tree texture_bind_group_layout"),
            });

        let tree_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &tree_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tree_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&tree_texture.sampler),
                },
            ],
            label: Some("tree diffuse_bind_group"),
        });

        let _color_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Color Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("color_shader.wgsl").into()),
        });

        let texture_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Texture Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("texture_shader.wgsl").into()),
        });

        let pentagon_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pentagon Render Pipeline Layout"),
                bind_group_layouts: &[&tree_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pentagon_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Pentagon Render Pipeline"),
                layout: Some(&pentagon_render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &texture_shader,
                    entry_point: "vs_main",
                    buffers: &[TexturedVertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &texture_shader,
                    entry_point: "fs_main",
                    targets: &[wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let star_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Star Render Pipeline Layout"),
                bind_group_layouts: &[&star_texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let star_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Star Render Pipeline"),
            layout: Some(&star_render_pipeline_layout),
            vertex: wgpu::VertexState {
                // module: &_color_shader,
                module: &texture_shader,
                entry_point: "vs_main",
                buffers: &[TexturedVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                // module: &_color_shader,
                module: &texture_shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let star_shape_buffers = ShapeBuffers {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Star Vertex Buffer"),
                contents: bytemuck::cast_slice(&star_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Star Index Buffer"),
                contents: bytemuck::cast_slice(star_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            num_vertices: star_vertices.len() as u32,
            num_indices: star_indices.len() as u32,
        };

        let pentagon_shape_buffers = ShapeBuffers {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pentagon Vertex Buffer"),
                contents: bytemuck::cast_slice(&pentagon_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pentagon Index Buffer"),
                contents: bytemuck::cast_slice(pentagon_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            num_vertices: pentagon_vertices.len() as u32,
            num_indices: pentagon_indices.len() as u32,
        };

        Self {
            surface,
            device,
            queue,
            config,
            size,

            current_mouse_pos: None,
            chosen_shape: ChosenShape::PENTAGON,

            star_render_pipeline,
            star_shape_buffers,
            star_diffuse_texture_bind_group: star_texture_bind_group,
            star_diffuse_texture: star_texture,

            pentagon_render_pipeline,
            pentagon_shape_buffers,
            pentagon_diffuse_texture_bind_group: tree_texture_bind_group,
            pentagon_diffuse_texture: tree_texture,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // Reconfigure the surface with the new size
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
        if let winit::event::WindowEvent::CursorMoved { position, .. } = event {
            self.current_mouse_pos = Some(*position);
        }
        false
    }

    fn update(&mut self) {}

    fn toggle_shape(&mut self) {
        self.chosen_shape = match &self.chosen_shape {
            ChosenShape::STAR => ChosenShape::PENTAGON,
            ChosenShape::PENTAGON => ChosenShape::STAR,
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        let clear_color = match self.current_mouse_pos {
            Some(pos) => wgpu::Color {
                r: to_srgb(pos.x as f32 / self.size.width as f32) as f64,
                g: to_srgb(pos.y as f32 / self.size.height as f32) as f64,
                b: 1.0,
                a: 1.0,
            },
            None => wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 1.0,
                a: 1.0,
            },
        };
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            let (shape_buffers, pipeline, bind_group) = match &self.chosen_shape {
                ChosenShape::STAR => (
                    &self.star_shape_buffers,
                    &self.star_render_pipeline,
                    Some(&self.star_diffuse_texture_bind_group),
                ),
                ChosenShape::PENTAGON => (
                    &self.pentagon_shape_buffers,
                    &self.pentagon_render_pipeline,
                    Some(&self.pentagon_diffuse_texture_bind_group),
                ),
            };
            render_pass.set_pipeline(pipeline);
            if let Some(bind_group) = bind_group {
                render_pass.set_bind_group(0, bind_group, &[]);
            }
            render_pass.set_vertex_buffer(0, shape_buffers.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                shape_buffers.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            render_pass.draw_indexed(0..shape_buffers.num_indices, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

async fn run() {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(
            FRAME_WIDTH as f64,
            FRAME_HEIGHT as f64,
        ))
        .with_title("David's window name")
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        // let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = winit::event_loop::ControlFlow::Wait;
        match event {
            winit::event::Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        *control_flow = winit::event_loop::ControlFlow::Exit
                    }
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            winit::event::Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            winit::event::Event::WindowEvent {
                event, window_id, ..
            } if window_id == window.id() => {
                if !state.input(&event) {
                    match event {
                        winit::event::WindowEvent::Resized(size) => {
                            state.resize(size);
                        }
                        winit::event::WindowEvent::ScaleFactorChanged {
                            new_inner_size, ..
                        } => {
                            state.resize(*new_inner_size);
                        }
                        winit::event::WindowEvent::CloseRequested
                        | winit::event::WindowEvent::KeyboardInput {
                            input:
                                winit::event::KeyboardInput {
                                    state: winit::event::ElementState::Pressed,
                                    virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = winit::event_loop::ControlFlow::Exit,
                        winit::event::WindowEvent::KeyboardInput {
                            input:
                                winit::event::KeyboardInput {
                                    state: winit::event::ElementState::Pressed,
                                    virtual_keycode: Some(winit::event::VirtualKeyCode::Space),
                                    ..
                                },
                            ..
                        } => state.toggle_shape(),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    });
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}
