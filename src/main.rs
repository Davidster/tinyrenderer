mod helpers;
mod mesh_component;
mod transform;

use std::borrow::Cow;

use helpers::*;
use mesh_component::*;
use transform::*;

const FRAME_WIDTH: i64 = 1920;
const FRAME_HEIGHT: i64 = 1080;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    current_mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &winit::window::Window) -> Self {
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

        Self {
            surface,
            device,
            queue,
            config,
            size,
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
                r: pos.x / self.size.width as f64,
                g: pos.y / self.size.height as f64,
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
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
