struct WindowMgr {
    event_loop: EventLoop<()>,
    surface: Arc<Surface<Window>>,
}

impl WindowMgr {
    fn init(base: &Base) -> Self {
        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            //.with_decorations(false)
            .with_transparent(true)
            .build_vk_surface(&event_loop, base.instance.clone())
            .unwrap();

        Self {
            event_loop,
            surface,
        }
    }
}

struct GraphMgr {
    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPass>,
    graph_pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

struct GraphStore {
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: Viewport,
    recreate_swapchain: bool,
}

impl GraphMgr {
    fn init(base: &Base, window_mgr: &WindowMgr, graph_store: &mut GraphStore) -> Self {
        let (swapchain, images) = Self::create_swapchain_and_images(base, window_mgr);
        let render_pass = Self::create_renderpass(&swapchain, base);
        let graph_pipeline = Self::create_graphics_pipeline(graph_store, &render_pass, base);
        let framebuffers = Self::window_size_dependent_setup(
            &images,
            render_pass.clone(),
            &mut graph_store.viewport,
        );
        let previous_frame_end = Self::create_previous_frame_end(&base);

        Self {
            swapchain,
            images,
            render_pass,
            graph_pipeline,
            framebuffers,
            previous_frame_end,
        }
    }

    fn create_swapchain_and_images(
        base: &Base,
        window_mgr: &WindowMgr,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let surface_capabilities = base
            .physical_device
            .surface_capabilities(&window_mgr.surface, Default::default())
            .unwrap();

        // Choosing the internal format that the images will have.
        let image_format = Some(
            base.physical_device
                .surface_formats(&window_mgr.surface, Default::default())
                .unwrap()[0]
                .0,
        );

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            base.device.clone(),
            window_mgr.surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,

                image_format,
                // The dimensions of the window, only used to initially setup the swapchain.
                // NOTE:
                // On some drivers the swapchain dimensions are specified by
                // `surface_capabilities.current_extent` and the swapchain size must use these
                // dimensions.
                // These dimensions are always the same as the window dimensions.
                //
                // However, other drivers don't specify a value, i.e.
                // `surface_capabilities.current_extent` is `None`. These drivers will allow
                // anything, but the only sensible value is the window
                // dimensions.
                //
                // Both of these cases need the swapchain to use the window dimensions, so we just
                // use that.
                image_extent: window_mgr.surface.window().inner_size().into(),

                image_usage: ImageUsage::color_attachment(),

                // The alpha mode indicates how the alpha value of the final image will behave. For
                // example, you can choose whether the window will be opaque or transparent.
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            },
        )
        .unwrap()
    }
    fn create_renderpass(swapchain: &Arc<Swapchain<Window>>, base: &Base) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(
            base.device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to
                    // be one of the types of the `vulkano::format` module (or alternatively one
                    // of your structs that implements the `FormatDesc` trait). Here we use the
                    // same format as the swapchain.
                    format: swapchain.image_format(),
                    // TODO:
                    samples: 1,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        )
        .unwrap()
    }

    fn create_graphics_pipeline(
        shaders: &GraphStore,
        render_pass: &Arc<RenderPass>,
        base: &Base,
    ) -> Arc<GraphicsPipeline> {
        GraphicsPipeline::start()
            // We need to indicate the layout of the vertices.
            .vertex_input_state(
                BuffersDefinition::new()    
                    .vertex::<Vertex>()
            )
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.
            .vertex_shader(shaders.vs.entry_point("main").unwrap(), ())
            // The content of the vertex buffer describes a list of triangles.
            .input_assembly_state(InputAssemblyState::new().topology(TOPOLOGY_TYPE))
            // Use a resizable viewport set to draw over the entire window
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            // See `vertex_shader`.
            .fragment_shader(shaders.fs.entry_point("main").unwrap(), ())
            // We have to indicate which subpass of which render pass this pipeline is going to be used
            // in. The pipeline will only be usable from this particular subpass.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
            .build(base.device.clone())
            .unwrap()
    }

    fn window_size_dependent_setup(
        images: &[Arc<SwapchainImage<Window>>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> Vec<Arc<Framebuffer>> {
        let dimensions = images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn recreate_framebuffer(&mut self, graph_store: &mut GraphStore) {
        self.framebuffers = Self::window_size_dependent_setup(
            &self.images,
            self.render_pass.clone(),
            &mut graph_store.viewport,
        );
    }

    fn create_previous_frame_end(base: &Base) -> Option<Box<dyn GpuFuture>> {
        Some(sync::now(base.device.clone()).boxed())
    }

    fn recreate_previous_frame_end(
        &mut self,
        graph_store: &mut GraphStore,
        device: Arc<Device>,
        future: Result<
            FenceSignalFuture<
                PresentFuture<
                    CommandBufferExecFuture<
                        JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture<Window>>,
                        PrimaryAutoCommandBuffer,
                    >,
                    Window,
                >,
            >,
            FlushError,
        >,
    ) {
        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                graph_store.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(device).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(device).boxed());
            }
        }
    }

    fn free_resources(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
    }
}