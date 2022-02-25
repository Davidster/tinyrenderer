mod helpers;
mod mesh_component;
mod transform;

use std::borrow::Cow;

use helpers::*;
use mesh_component::*;
use transform::*;

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
    render_pipeline: wgpu::RenderPipeline,
    pentagon_shape_buffers: ShapeBuffers,
    star_shape_buffers: ShapeBuffers,
    chosen_shape: ChosenShape,
    current_mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,
}

type VertexPosition = [f32; 3];
type VertexColor = [f32; 3];

// main.rs
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: VertexPosition,
    color: VertexColor,
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

impl Vertex {
    const _ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        // verbose version:
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<VertexPosition>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }

        // concise version:
        // wgpu::VertexBufferLayout {
        //     array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        //     step_mode: wgpu::VertexStepMode::Vertex,
        //     attributes: &Self::_ATTRIBS,
        // }
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
        // TODO: make a perfect start of David in a square
        // let l1 = 1.0 - ((1.0 - 0.5 * 0.5) as f32).sqrt();
        // let l12 = 0.5 - l1;
        // let l2 = (l1 / (1.0 - l1)) * 0.5;
        // let l22 = (0.5 / (1.0 - l1)) * 0.5;
        // let star_vertices: Vec<Vertex> = vec![
        //     [-0.5, l12],
        //     [-0.5, -l12],
        //     [0.5, -l12],
        //     [0.5, l12],
        //     [0.0, 0.5],
        //     [0.0, -0.5],
        //     [-l2, l12],
        //     [l2, l12],
        //     [l2, -l12],
        //     [-l2, -l12],
        //     [-l22, 0.0],
        //     [l22, 0.0],
        // ]
        let star_vertices: Vec<Vertex> = vec![
            [-0.5, 0.25],
            [-0.5, -0.25],
            [0.5, -0.25],
            [0.5, 0.25],
            [0.0, 0.5],
            [0.0, -0.5],
            [-0.161115, 0.25],
            [0.161115, 0.25],
            [0.161115, -0.25],
            [-0.161115, -0.25],
            [-0.33333335, 0.0],
            [0.33333335, 0.0],
        ]
        .iter()
        .map(|pos| Vertex {
            position: [pos[0], pos[1], 0.0],
            color: star_color.clone(),
        })
        .collect();

        let star_indices: &[u16] = &[
            0, 10, 6, 4, 6, 7, 7, 11, 3, 11, 8, 2, 8, 9, 5, 9, 10, 1, 6, 10, 9, 7, 8, 11, 6, 9, 7,
            7, 9, 8,
        ];

        let pentagon_color = [to_srgb(0.5), 0.0, to_srgb(0.5)];
        let pentagon_vertices: Vec<Vertex> = vec![
            [-0.0868241, 0.49240386],
            [-0.49513406, 0.06958647],
            [-0.21918549, -0.44939706],
            [0.35966998, -0.3473291],
            [0.44147372, 0.2347359],
        ]
        .iter()
        .map(|pos| Vertex {
            position: [pos[0], pos[1], 0.0],
            color: pentagon_color.clone(),
        })
        .collect();
        // let pentagon_vertices: &[Vertex] = &[
        //     Vertex {
        //         position: [-0.0868241, 0.49240386, 0.0],
        //         color: pentagon_color.clone(),
        //     },
        //     Vertex {
        //         position: [-0.49513406, 0.06958647, 0.0],
        //         color: pentagon_color.clone(),
        //     },
        //     Vertex {
        //         position: [-0.21918549, -0.44939706, 0.0],
        //         color: pentagon_color.clone(),
        //     },
        //     Vertex {
        //         position: [0.35966998, -0.3473291, 0.0],
        //         color: pentagon_color.clone(),
        //     },
        //     Vertex {
        //         position: [0.44147372, 0.2347359, 0.0],
        //         color: pentagon_color.clone(),
        //     },
        // ];

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

        let mut config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &config);

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
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
                contents: bytemuck::cast_slice(star_vertices.as_slice()),
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
                contents: bytemuck::cast_slice(pentagon_vertices.as_slice()),
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
            render_pipeline,
            star_shape_buffers,
            pentagon_shape_buffers,
            chosen_shape: ChosenShape::STAR,
            current_mouse_pos: None,
        }

        // wgpu docs say that wgsl is not supported on the web, but this seems to no longer
        // be the case once v0.13.0 is released: https://github.com/gfx-rs/wgpu/pull/2315
        // let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        //     label: None,
        //     source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        // });

        // let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //     label: None,
        //     bind_group_layouts: &[],
        //     push_constant_ranges: &[],
        // });

        // let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        //     label: None,
        //     layout: Some(&pipeline_layout),
        //     vertex: wgpu::VertexState {
        //         module: &shader,
        //         entry_point: "vs_main",
        //         buffers: &[],
        //     },
        //     fragment: Some(wgpu::FragmentState {
        //         module: &shader,
        //         entry_point: "fs_main",
        //         targets: &[swapchain_format.into()],
        //     }),
        //     primitive: wgpu::PrimitiveState::default(),
        //     depth_stencil: None,
        //     multisample: wgpu::MultisampleState::default(),
        //     multiview: None,
        // });
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

            render_pass.set_pipeline(&self.render_pipeline);
            let shape_buffers = match &self.chosen_shape {
                ChosenShape::STAR => &self.star_shape_buffers,
                ChosenShape::PENTAGON => &self.pentagon_shape_buffers,
            };
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
        // .with_inner_size(winit::dpi::LogicalSize::new(
        //     FRAME_WIDTH as f64,
        //     FRAME_HEIGHT as f64,
        // ))
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
