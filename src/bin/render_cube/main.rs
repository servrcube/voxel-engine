mod base;
mod graphics;
use std::time::Instant;
use base::{Base, BaseOptions};
use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use std::sync::Arc;
use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolChunk, BufferAccess, BufferContents, BufferUsage,
        CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess,
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
        PrimaryAutoCommandBuffer, PrimaryCommandBuffer, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    memory::MemoryPool,
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, PresentFuture, Surface, Swapchain,
        SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FenceSignalFuture, FlushError, GpuFuture, JoinFuture},
};
use vulkano_win::VkSurfaceBuild;

use graphics::{GraphMgr, GraphStore, WindowMgr, Vertex};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    // The first step of any Vulkan program is to create an instance.
    //
    // When we create an instance, we have to pass a list of extensions that we want to enable.
    //
    // All the window-drawing functionalities are part of non-core extensions that we need
    // to enable manually. To do so, we ask the `vulkano_win` crate for the list of extensions
    // required to draw to a window.

    // Now creating the instance.
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: vulkano_win::required_extensions(),
        ..Default::default()
    })
    .unwrap();

    let mut base = Base::default(
        &instance,
        DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        },
    );

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/bin/render_cube/vertShader.vert",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/bin/render_cube/fragShader.frag",
        }
    }

    let vs = vs::load(base.device.clone()).unwrap();
    let fs = fs::load(base.device.clone()).unwrap();

    let queue = base.queues.next().unwrap();
    let window_mgr = WindowMgr::init(&base);
    let mut graph_store = GraphStore {
        vs,
        fs,
        recreate_swapchain: false,
    };
    let mut graph_mgr = GraphMgr::init(&base, &window_mgr, &mut graph_store);

    let buffer_pool_verteices = CpuBufferPool::new(base.device.clone(), BufferUsage::all());
    let buffer_pool_uniforms = CpuBufferPool::new(base.device.clone(), BufferUsage::all());
    let buffer_pool_indices = CpuBufferPool::new(base.device.clone(), BufferUsage::all());

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.

    let rotation_start = Instant::now();

    window_mgr.event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                graph_store.recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                graph_mgr.free_resources();

                // Whenever the window resizes we need to recreate everything dependent on the window size.
                // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
                if graph_store.recreate_swapchain {
                    // Get the new dimensions of the window.

                    let (new_swapchain, new_images) =
                        match graph_mgr.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: window_mgr.surface.window().inner_size().into(),
                            ..graph_mgr.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            // This error tends to happen when the user is manually resizing the window.
                            // Simply restarting the loop is the easiest way to fix this issue.
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    graph_mgr.swapchain = new_swapchain;
                    graph_mgr.images = new_images;
                    // Because framebuffers contains an Arc on the old swapchain, we need to
                    // recreate framebuffers as well.
                    graph_mgr.recreate_framebuffer(&base.device);
                    graph_mgr.recreate_pipline(&graph_store, &base.device);
                    graph_store.recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
                // no image is available (which happens if you submit draw commands too quickly), then the
                // function will block.
                // This operation returns the index of the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional timeout
                // after which the function call will return an error.
                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(graph_mgr.swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            graph_store.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                // will still work, but it may not display correctly. With some drivers this can be when
                // the window resizes, but it may not cause the swapchain to become out of date.
                if suboptimal {
                    graph_store.recreate_swapchain = true;
                }

                // Specify the color to clear the framebuffer with i.e. blue
                
                
                let vertices = [
                    Vertex {
                        position: [-20.0, -20.0, 50.0],
                    },
                    Vertex {
                        position: [-20.0, 20.0, 50.0],
                    },
                    Vertex {
                        position: [20.0, 20.0, 50.0],
                    },
                    Vertex {
                        position: [20.0, -20.0, 50.0],
                    },

                    Vertex {
                        position: [-20.0, -20.0, 20.0],
                    },
                    Vertex {
                        position: [-20.0, 20.0, 20.0],
                    },
                    Vertex {
                        position: [20.0, 20.0, 20.0],
                    },
                    Vertex {
                        position: [20.0, -20.0, 20.0],
                    },
                ];

                let indices: Vec<u16> = vec![
                    0,1,2,
                    0,2,3,

                    7,5,4,
                    7,5,6
                ];

                let uniform_data = {

                    let elapsed = rotation_start.elapsed();
                    let rotation =
                        elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                    let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                    let aspect_ratio =
                        graph_mgr.swapchain.image_extent()[0] as f32 / graph_mgr.swapchain.image_extent()[1] as f32;
                    let proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect_ratio,
                        0.01,
                        100.0,
                    );
                    let view = Matrix4::look_at_rh(
                        Point3::new(0.3, 0.3, 1.0),
                        Point3::new(0.0, 0.0, 0.0),
                        Vector3::new(0.0, -1.0, 0.0),
                    );
                    let scale = Matrix4::from_scale(0.01);

                    vs::ty::Data {
                        world: Matrix4::from(rotation).into(),
                        view: (view * scale)    .into(),
                        proj: proj.into(),
                    }

                };


                let vertex_buffer = buffer_pool_verteices.chunk(vertices).unwrap();
                let indices = buffer_pool_indices.chunk(indices).unwrap();
                let uniforms = buffer_pool_uniforms.next(uniform_data).unwrap();
                
                let layout = graph_mgr
                    .graph_pipeline
                    .layout()
                    .set_layouts()
                    .get(0)
                    .unwrap();
                let set = PersistentDescriptorSet::new(
                    layout.clone(),
                    [WriteDescriptorSet::buffer(0, uniforms)],
                )
                .unwrap();

                // In order to draw, we have to build a *command buffer*. The command buffer object holds
                // the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to be
                // optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The command
                // buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    base.device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();


                builder
                    // Before we can draw, we have to *enter a render pass*. There are two methods to do
                    // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
                    // not covered here.
                    //
                    // The third parameter builds the list of values to clear the attachments with. The API
                    // is similar to the list of attachments when building the framebuffers, except that
                    // only the attachments that use `load: Clear` appear in the list.
                    .begin_render_pass(
                        graph_mgr.framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        vec![[1.0, 1.0, 1.0, 1.0].into(), 1f32.into(),],
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass. We add a draw command.
                    //
                    // The last two parameters contain the list of resources to pass to the shaders.
                    // Since we used an `EmptyPipeline` object, the objects have to be `()`.
                    .bind_pipeline_graphics(graph_mgr.graph_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        graph_mgr.graph_pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_index_buffer(indices.clone())
                    .draw_indexed(indices.len() as u32, 1, 0, 0, 0)
                    .unwrap()
                    // We leave the render pass by calling `draw_end`. Note that if we had multiple
                    // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
                    // next subpass.
                    .end_render_pass()
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = graph_mgr
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to show it on
                    // the screen, we have to *present* the image by calling `present`.
                    //
                    // This function does not actually present the image immediately. Instead it submits a
                    // present command at the end of the queue. This means that it will only be presented once
                    // the GPU has finished executing the command buffer that draws the triangle.
                    .then_swapchain_present(queue.clone(), graph_mgr.swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                graph_mgr.recreate_previous_frame_end(
                    &mut graph_store,
                    base.device.clone(),
                    future,
                );
            }
            _ => (),
        }
    });
}
