use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    instance::Instance,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::{ShaderCreationError, ShaderModule},
    sync::{self, GpuFuture},
};

use colored_output::*;

use std::iter::{ExactSizeIterator, Iterator};
use std::result::Result;
use std::sync::Arc;

// this is used to combine two traits together,
// otherwise it would not work
// Lesson -> to combine two traits as a object make new trait
/// Trait for a Queue Trait and a Exact size iterator
pub trait QueuesTrait: ExactSizeIterator + Iterator<Item = Arc<Queue>> {}
impl<T> QueuesTrait for T where T: ExactSizeIterator + Iterator<Item = Arc<Queue>> {}

/// trait load used in junction with a mod shader code
pub trait Load {
    fn load(device: Arc<Device>) -> Result<Arc<ShaderModule>, ShaderCreationError>;
}

/// A struct to store all the required values to make any pipline
pub struct Base<'a> {
    pub instance: Arc<Instance>,
    pub physical_device: PhysicalDevice<'a>,
    pub available_physical_devices: Vec<PhysicalDevice<'a>>,
    pub device: Arc<Device>,
    pub queues: Box<dyn QueuesTrait>,
}

/// Passed to the Pipeline base
///
/// Provides all user defined possible specializasions
pub struct BaseOptions<'a> {
    pub device_extensions: DeviceExtensions,
    pub physical_device: PhysicalDevice<'a>,
    pub queue_family: QueueFamily<'a>,
}

impl<'a> Base<'a> {
    /// initializes a Pipeline base
    ///
    /// accepts a instance manually created
    pub fn init(instance: &'a Arc<Instance>, opts: BaseOptions<'a>) -> Self {
        let available_physical_devices =
            Self::get_available_devices(instance, &opts.device_extensions);

        println!();
        info!(
            "Using device: {} (type: {:?})",
            opts.physical_device.properties().device_name,
            opts.physical_device.properties().device_type
        );

        info!("Available Devices:\n");
        available_physical_devices.iter().for_each(|&p| {
            eprintln!(
                "\t {} {} \n\t {} {:?}\n",
                "Device:".blue().dimmed().bold(),
                p.properties().device_name,
                "Type:".blue().dimmed().bold(),
                p.properties().device_type
            );
        });
        eprintln!();

        // Now initializing the device.
        let (device, queues) = Device::new(
            opts.physical_device,
            DeviceCreateInfo {
                enabled_extensions: opts
                    .physical_device
                    .required_extensions()
                    .union(&opts.device_extensions),
                queue_create_infos: vec![QueueCreateInfo::family(opts.queue_family)],
                ..Default::default()
            },
        )
        .unwrap();

        info!("Successfully initalized Pipeline");

        Self {
            instance: instance.clone(),
            physical_device: opts.physical_device,
            available_physical_devices,
            device,
            queues: Box::from(queues),
        }
    }

    /// Gets the physical device by a pre-selected hierachy
    ///
    /// Also returns a queue family for further usage
    pub fn get_device_by_hierachy(
        instance: &'a Arc<Instance>,
        device_extensions: &DeviceExtensions,
    ) -> (PhysicalDevice<'a>, QueueFamily<'a>) {
        PhysicalDevice::enumerate(instance)
            .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
            .filter_map(|p| {
                // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
                // that supports compute operations.
                p.queue_families()
                    .find(|&q| q.supports_compute())
                    .map(|q| (p, q))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            })
            .unwrap()
    }

    /// gets all available Physical devices
    pub fn get_available_devices(
        instance: &'a Arc<Instance>,
        device_extensions: &DeviceExtensions,
    ) -> Vec<PhysicalDevice<'a>> {
        PhysicalDevice::enumerate(&instance)
            .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
            .filter_map(|p| {
                // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
                // that supports compute operations.
                p.queue_families()
                    .find(|&q| q.supports_compute())
                    .map(|q| (p, q))
            })
            .map(|(p, _)| p)
            .collect()
    }

    /// passes default parameters for PiplineBase
    pub fn default(instance: &'a Arc<Instance>, device_extensions: DeviceExtensions) -> Self {
        let device_constr = Self::get_device_by_hierachy(instance, &device_extensions);

        let pipeline_options = BaseOptions {
            device_extensions,
            physical_device: device_constr.0,
            queue_family: device_constr.1,
        };

        Base::init(instance, pipeline_options)
    }

    fn get_device_and_queue(
        opts: &BaseOptions,
    ) -> (
        Arc<Device>,
        impl ExactSizeIterator + Iterator<Item = Arc<Queue>>,
    ) {
        Device::new(
            opts.physical_device,
            DeviceCreateInfo {
                enabled_extensions: opts
                    .physical_device
                    .required_extensions()
                    .union(&opts.device_extensions),
                queue_create_infos: vec![QueueCreateInfo::family(opts.queue_family)],
                ..Default::default()
            },
        )
        .unwrap()
    }
}
