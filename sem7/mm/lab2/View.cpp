#include "View.hpp"
#include <stdio.h>          // printf, fprintf
#include <stdlib.h>         // abort
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>


// Volk headers
#ifdef IMGUI_IMPL_VULKAN_USE_VOLK
#define VOLK_IMPLEMENTATION
#include <volk.h>
#endif

//#define APP_USE_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define APP_USE_VULKAN_DEBUG_REPORT
#endif

// Data
static VkAllocationCallbacks* g_Allocator = nullptr;
static VkInstance               g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice         g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice                 g_Device = VK_NULL_HANDLE;
static uint32_t                 g_QueueFamily = (uint32_t)-1;
static VkQueue                  g_Queue = VK_NULL_HANDLE;
static VkDebugReportCallbackEXT g_DebugReport = VK_NULL_HANDLE;
static VkPipelineCache          g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool         g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint32_t                 g_MinImageCount = 2;
static bool                     g_SwapChainRebuild = false;

static void check_vk_result(VkResult err)
{
	if (err == 0)
		return;
	fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
	if (err < 0)
		abort();
}

#ifdef APP_USE_VULKAN_DEBUG_REPORT
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
	(void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
	fprintf(stderr, "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n", objectType, pMessage);
	return VK_FALSE;
}
#endif // APP_USE_VULKAN_DEBUG_REPORT

static bool IsExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension)
{
	for (const VkExtensionProperties& p : properties)
		if (strcmp(p.extensionName, extension) == 0)
			return true;
	return false;
}

static VkPhysicalDevice SetupVulkan_SelectPhysicalDevice()
{
	uint32_t gpu_count;
	VkResult err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, nullptr);
	check_vk_result(err);
	IM_ASSERT(gpu_count > 0);

	ImVector<VkPhysicalDevice> gpus;
	gpus.resize(gpu_count);
	err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, gpus.Data);
	check_vk_result(err);

	// If a number >1 of GPUs got reported, find discrete GPU if present, or use first one available. This covers
	// most common cases (multi-gpu/integrated+dedicated graphics). Handling more complicated setups (multiple
	// dedicated GPUs) is out of scope of this sample.
	for (VkPhysicalDevice& device : gpus)
	{
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(device, &properties);
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			return device;
	}

	// Use first GPU (Integrated) is a Discrete one is not available.
	if (gpu_count > 0)
		return gpus[0];
	return VK_NULL_HANDLE;
}

static void SetupVulkan(ImVector<const char*> instance_extensions)
{
	VkResult err;
#ifdef IMGUI_IMPL_VULKAN_USE_VOLK
	volkInitialize();
#endif

	// Create Vulkan Instance
	{
		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

		// Enumerate available extensions
		uint32_t properties_count;
		ImVector<VkExtensionProperties> properties;
		vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
		properties.resize(properties_count);
		err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.Data);
		check_vk_result(err);

		// Enable required extensions
		if (IsExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
			instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
		if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
		{
			instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
			create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
		}
#endif

		// Enabling validation layers
#ifdef APP_USE_VULKAN_DEBUG_REPORT
		const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
		create_info.enabledLayerCount = 1;
		create_info.ppEnabledLayerNames = layers;
		instance_extensions.push_back("VK_EXT_debug_report");
#endif

		// Create Vulkan Instance
		create_info.enabledExtensionCount = (uint32_t)instance_extensions.Size;
		create_info.ppEnabledExtensionNames = instance_extensions.Data;
		err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
		check_vk_result(err);
#ifdef IMGUI_IMPL_VULKAN_USE_VOLK
		volkLoadInstance(g_Instance);
#endif

		// Setup the debug report callback
#ifdef APP_USE_VULKAN_DEBUG_REPORT
		auto f_vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkCreateDebugReportCallbackEXT");
		IM_ASSERT(f_vkCreateDebugReportCallbackEXT != nullptr);
		VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
		debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		debug_report_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
		debug_report_ci.pfnCallback = debug_report;
		debug_report_ci.pUserData = nullptr;
		err = f_vkCreateDebugReportCallbackEXT(g_Instance, &debug_report_ci, g_Allocator, &g_DebugReport);
		check_vk_result(err);
#endif
	}

	// Select Physical Device (GPU)
	g_PhysicalDevice = SetupVulkan_SelectPhysicalDevice();

	// Select graphics queue family
	{
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, nullptr);
		VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
		vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, queues);
		for (uint32_t i = 0; i < count; i++)
			if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				g_QueueFamily = i;
				break;
			}
		free(queues);
		IM_ASSERT(g_QueueFamily != (uint32_t)-1);
	}

	// Create Logical Device (with 1 queue)
	{
		ImVector<const char*> device_extensions;
		device_extensions.push_back("VK_KHR_swapchain");

		// Enumerate physical device extension
		uint32_t properties_count;
		ImVector<VkExtensionProperties> properties;
		vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, nullptr);
		properties.resize(properties_count);
		vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, properties.Data);
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
		if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
			device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

		const float queue_priority[] = { 1.0f };
		VkDeviceQueueCreateInfo queue_info[1] = {};
		queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[0].queueFamilyIndex = g_QueueFamily;
		queue_info[0].queueCount = 1;
		queue_info[0].pQueuePriorities = queue_priority;
		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
		create_info.pQueueCreateInfos = queue_info;
		create_info.enabledExtensionCount = (uint32_t)device_extensions.Size;
		create_info.ppEnabledExtensionNames = device_extensions.Data;
		err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
		check_vk_result(err);
		vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
	}

	// Create Descriptor Pool
	// The example only requires a single combined image sampler descriptor for the font image and only uses one descriptor set (for that)
	// If you wish to load e.g. additional textures you may need to alter pools sizes.
	{
		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
		};
		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = 1;
		pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;
		err = vkCreateDescriptorPool(g_Device, &pool_info, g_Allocator, &g_DescriptorPool);
		check_vk_result(err);
	}
}

// All the ImGui_ImplVulkanH_XXX structures/functions are optional helpers used by the demo.
// Your real engine/app may not use them.
static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height)
{
	wd->Surface = surface;

	// Check for WSI support
	VkBool32 res;
	vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily, wd->Surface, &res);
	if (res != VK_TRUE)
	{
		fprintf(stderr, "Error no WSI support on physical device 0\n");
		exit(-1);
	}

	// Select Surface Format
	const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
	const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
	wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(g_PhysicalDevice, wd->Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

	// Select Present Mode
#ifdef APP_UNLIMITED_FRAME_RATE
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
#else
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
#endif
	wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(g_PhysicalDevice, wd->Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));
	//printf("[vulkan] Selected PresentMode = %d\n", wd->PresentMode);

	// Create SwapChain, RenderPass, Framebuffer, etc.
	IM_ASSERT(g_MinImageCount >= 2);
	ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, wd, g_QueueFamily, g_Allocator, width, height, g_MinImageCount);
}

static void CleanupVulkan()
{
	vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

#ifdef APP_USE_VULKAN_DEBUG_REPORT
	// Remove the debug report callback
	auto f_vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkDestroyDebugReportCallbackEXT");
	f_vkDestroyDebugReportCallbackEXT(g_Instance, g_DebugReport, g_Allocator);
#endif // APP_USE_VULKAN_DEBUG_REPORT

	vkDestroyDevice(g_Device, g_Allocator);
	vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow()
{
	ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData, g_Allocator);
}

static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
{
	VkResult err;

	VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
	VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
	if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
	{
		g_SwapChainRebuild = true;
		return;
	}
	check_vk_result(err);

	ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
	{
		err = vkWaitForFences(g_Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);    // wait indefinitely instead of periodically checking
		check_vk_result(err);

		err = vkResetFences(g_Device, 1, &fd->Fence);
		check_vk_result(err);
	}
	{
		err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
		check_vk_result(err);
		VkCommandBufferBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
		check_vk_result(err);
	}
	{
		VkRenderPassBeginInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		info.renderPass = wd->RenderPass;
		info.framebuffer = fd->Framebuffer;
		info.renderArea.extent.width = wd->Width;
		info.renderArea.extent.height = wd->Height;
		info.clearValueCount = 1;
		info.pClearValues = &wd->ClearValue;
		vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// Record dear imgui primitives into command buffer
	ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

	// Submit command buffer
	vkCmdEndRenderPass(fd->CommandBuffer);
	{
		VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &image_acquired_semaphore;
		info.pWaitDstStageMask = &wait_stage;
		info.commandBufferCount = 1;
		info.pCommandBuffers = &fd->CommandBuffer;
		info.signalSemaphoreCount = 1;
		info.pSignalSemaphores = &render_complete_semaphore;

		err = vkEndCommandBuffer(fd->CommandBuffer);
		check_vk_result(err);
		err = vkQueueSubmit(g_Queue, 1, &info, fd->Fence);
		check_vk_result(err);
	}
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd)
{
	if (g_SwapChainRebuild)
		return;
	VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	VkPresentInfoKHR info = {};
	info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	info.waitSemaphoreCount = 1;
	info.pWaitSemaphores = &render_complete_semaphore;
	info.swapchainCount = 1;
	info.pSwapchains = &wd->Swapchain;
	info.pImageIndices = &wd->FrameIndex;
	VkResult err = vkQueuePresentKHR(g_Queue, &info);
	if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
	{
		g_SwapChainRebuild = true;
		return;
	}
	check_vk_result(err);
	wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount; // Now we can use the next set of semaphores
}

View::View()
{
// Setup SDL
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
	{
		printf("Error: %s\n", SDL_GetError());
		std::abort();
	}

	// From 2.0.18: Enable native IME.
#ifdef SDL_HINT_IME_SHOW_UI
	SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

	// Create window with Vulkan graphics context
	window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI);
	window = SDL_CreateWindow("Queueing Model", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
	if (window == nullptr)
	{
		printf("Error: SDL_CreateWindow(): %s\n", SDL_GetError());
		std::abort();
	}

	extensions_count = 0;
	SDL_Vulkan_GetInstanceExtensions(window, &extensions_count, nullptr);
	extensions.resize(extensions_count);
	SDL_Vulkan_GetInstanceExtensions(window, &extensions_count, extensions.Data);
	SetupVulkan(extensions);

	// Create Window Surface
	if (SDL_Vulkan_CreateSurface(window, g_Instance, &surface) == 0)
	{
		printf("Failed to create Vulkan surface.\n");
		std::abort();
	}

	// Create Framebuffers
	SDL_GetWindowSize(window, &w, &h);
	wd = &g_MainWindowData;
	SetupVulkanWindow(wd, surface, w, h);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); (void)io;
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	ImGui_ImplSDL2_InitForVulkan(window);
	init_info = {};
	init_info.Instance = g_Instance;
	init_info.PhysicalDevice = g_PhysicalDevice;
	init_info.Device = g_Device;
	init_info.QueueFamily = g_QueueFamily;
	init_info.Queue = g_Queue;
	init_info.PipelineCache = g_PipelineCache;
	init_info.DescriptorPool = g_DescriptorPool;
	init_info.RenderPass = wd->RenderPass;
	init_info.Subpass = 0;
	init_info.MinImageCount = g_MinImageCount;
	init_info.ImageCount = wd->ImageCount;
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.Allocator = g_Allocator;
	init_info.CheckVkResultFn = check_vk_result;
	ImGui_ImplVulkan_Init(&init_info);

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
	// - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
	// - Read 'docs/FONTS.md' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	//io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != nullptr);

	// Main loop
	done = false;
}

View::~View()
{
	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	CleanupVulkanWindow();
	CleanupVulkan();

	SDL_DestroyWindow(window);
	SDL_Quit();
}


void View::draw()
{
	processEvent();
	newFrame();
	drawWidgets();
	render();
}

bool View::isDone() const
{
	return done;
}

void View::processEvent()
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		ImGui_ImplSDL2_ProcessEvent(&event);
		if (event.type == SDL_QUIT)
			done = true;
		if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
			done = true;
	}
}

void View::newFrame()
{
	if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
	{
		SDL_Delay(10);
		return;
	}

	// Resize swap chain?
	int fb_width, fb_height;
	SDL_GetWindowSize(window, &fb_width, &fb_height);
	if (fb_width > 0 && fb_height > 0 && (g_SwapChainRebuild || g_MainWindowData.Width != fb_width || g_MainWindowData.Height != fb_height))
	{
		ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
		ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, g_QueueFamily, g_Allocator, fb_width, fb_height, g_MinImageCount);
		g_MainWindowData.FrameIndex = 0;
		g_SwapChainRebuild = false;
	}

	// Start the Dear ImGui frame
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
}

void View::render()
{
	ImGui::Render();
	ImDrawData* draw_data = ImGui::GetDrawData();
	const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
	if (!is_minimized)
	{
		wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
		wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
		wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
		wd->ClearValue.color.float32[3] = clear_color.w;
		FrameRender(wd, draw_data);
		FramePresent(wd);
	}
}

void View::drawWidgets()
{
	// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
	if (show_demo_window)
		ImGui::ShowDemoWindow(&show_demo_window);

	updateUIData();

	if (is_model_running && !isConcurency)
	{
		model.nextStep();
	}

	drawSystemVisualisation();

	drawRightPanel();
}

void View::drawRightPanel()
{
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.65f, 0.58f, 0.50f, 1.00f));
	ImGui::Begin("Right Window", &show_right_window, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoResize | ImGuiWindowFlags_UnsavedDocument);
	ImGui::SetWindowPos(ImVec2(w - ImGui::GetWindowWidth(), 0));
	ImGui::SetWindowSize(ImVec2(250, h));

	float input_width = 100.0f;

	ImGuiStyle& style = ImGui::GetStyle();
	style.Colors[ImGuiCol_PopupBg] = ImVec4(0.7f, 0.6f, 0.5f, 1.0f);
	style.Colors[ImGuiCol_FrameBg] = ImVec4(0.31f, 0.26f, 0.19f, 1.0f);

	ImGui::SetNextItemWidth(input_width);
	ImGui::InputFloat("Simulation Time", &simulation_time, .0f, .0f, "%.3g");
	ImGui::SetItemTooltip("Duration of the simulation in minutes.");

	ImGui::SetNextItemWidth(input_width);
	ImGui::InputInt("Number of lines", &num_lines);
	ImGui::SetItemTooltip(
		"Number of lines in the system. Increasing\n"
		"this parameter helps to reduce the number\n"
		"of rejected applications."
	);

	ImGui::SetNextItemWidth(input_width);
	ImGui::InputInt("Buffer capacity", &buffer_capacity);
	ImGui::SetItemTooltip(
		"Capacity of the buffer where unaccepted\n"
		"requests are sent. Increasing this\n"
		"parameter helps to reduce the number of\n"
		"rejected applications."
	);

	ImGui::SetNextItemWidth(input_width);
	ImGui::InputFloat("Arrival rate", &arrival_rate, .0f, .0f, "%.3g");
	ImGui::SetItemTooltip(
		"Affects the average time between\n"
		"requests (lambda). Increasing this\n"
		"parameter contributes to an increase\n"
		"in line congestion and the number\n"
		"of rejected applications."
	);

	ImGui::SetNextItemWidth(input_width);
	ImGui::InputFloat("Service rate", &reverse_service_time_mean, .0f, .0f, "%.3g");
	ImGui::SetItemTooltip(
		"Affects the average application processing\n"
		"time (beta). Increasing this parameter\n"
		"helps to reduce line congestion and the\n"
		"number of rejected applications."
	);

	ImGui::SetNextItemWidth(input_width);
	ImGui::InputFloat("Failure chance", &failure_chance, .0f, .0f, "%.3g");
	ImGui::SetItemTooltip(
		"Chance of line failure after processing\n"
		"the application. Increasing this parameter\n"
		"contributes to an increase in line\n"
		"congestion and the number of rejected applications."
	);

	ImGui::SetNextItemWidth(input_width);
	ImGui::InputFloat("Recovery rate", &recovery_rate, .0f, .0f, "%.3g");
	ImGui::SetItemTooltip(
		"Affects the average line recovery time. Increasing\n"
		"this parameter contributes to an increase in line\n"
		"congestion and the number of rejected applications."
	);

	if (is_model_running)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.0f, 0.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.4f, 0.4f, 1.0f));
		if (ImGui::Button("Stop Model", ImVec2(100, 30)))
		{
			std::lock_guard<std::mutex> lock(model_mutex);
			model.stopSimulation();
			is_model_running = false;
		}
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.7f, 0.4f, 1.0f));
		if (ImGui::Button("Start Model", ImVec2(100, 30)))
		{
			try
			{
				startSimulation();
				is_model_running = true;
			}
			catch (std::exception& e)
			{
				show_error_popup = true;
				ImGui::OpenPopup("Error");
				lastError = e.what();
			}
		}
		if (!is_model_running)
		{
			ImGui::SameLine();
			ImGui::Checkbox("Concurency", &isConcurency);
		}
	}
	ImGui::PopStyleColor(3);
	if (show_error_popup)
	{
		if (ImGui::BeginPopupModal("Error", NULL, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::Text("%s", lastError.c_str());
			if (ImGui::Button("OK", ImVec2(120, 0)))
			{
				show_error_popup = false;
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
	}

	// Output
	char effectivity_str[50];
	ImGui::SetNextItemWidth(input_width);
	snprintf(effectivity_str, sizeof(effectivity_str), "%.10g", effectivity);
	ImGui::InputText("Effectivity", effectivity_str, IM_ARRAYSIZE(effectivity_str), ImGuiInputTextFlags_ReadOnly);

	char arrivals_str[50];
	ImGui::SetNextItemWidth(input_width);
	snprintf(arrivals_str, sizeof(arrivals_str), "%d", arrivals_count);
	ImGui::InputText("Total arrivals", arrivals_str, IM_ARRAYSIZE(arrivals_str), ImGuiInputTextFlags_ReadOnly);

	char failures_str[50];
	ImGui::SetNextItemWidth(input_width);
	snprintf(failures_str, sizeof(failures_str), "%d", failures_count);
	ImGui::InputText("Total failures", failures_str, IM_ARRAYSIZE(failures_str), ImGuiInputTextFlags_ReadOnly);

	char rejected_str[50];
	ImGui::SetNextItemWidth(input_width);
	snprintf(rejected_str, sizeof(rejected_str), "%d", rejected_count);
	ImGui::InputText("Rejected", rejected_str, IM_ARRAYSIZE(rejected_str), ImGuiInputTextFlags_ReadOnly);

	char busy_lines_str[50];
	ImGui::SetNextItemWidth(input_width);
	snprintf(busy_lines_str, sizeof(busy_lines_str), "%d", num_busy_lines);
	ImGui::InputText("Busy lines", busy_lines_str, IM_ARRAYSIZE(busy_lines_str), ImGuiInputTextFlags_ReadOnly);

	char disabled_lines_str[50];
	ImGui::SetNextItemWidth(input_width);
	snprintf(disabled_lines_str, sizeof(disabled_lines_str), "%d", num_disabled_lines);
	ImGui::InputText("Disabled lines", disabled_lines_str, IM_ARRAYSIZE(disabled_lines_str), ImGuiInputTextFlags_ReadOnly);

	char buffer_usage_str[50];
	ImGui::SetNextItemWidth(input_width);
	snprintf(buffer_usage_str, sizeof(buffer_usage_str), "%d", buffer_usage);
	ImGui::InputText("Buffer usage", buffer_usage_str, IM_ARRAYSIZE(buffer_usage_str), ImGuiInputTextFlags_ReadOnly);

	ImGui::End();
	ImGui::PopStyleColor();
}


void View::drawSystemVisualisation()
{
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
	ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.96f, 0.90f, 0.80f, 1.00f));
	ImGui::Begin("Moving Point", &show_animation_window, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_UnsavedDocument);

	updateIntervals();

	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	for (size_t i = 0; i != lines.size(); ++i)
	{
		draw_list->AddLine(lines[i].first, lines[i].second, IM_COL32_BLACK, 1.0f);
		std::string label = std::to_string(i);
		draw_list->AddText(lines[i].first, IM_COL32_BLACK, label.c_str());
	}

	for (const auto& i : busy_intervals)
	{
		draw_list->AddLine(i.first, i.second, IM_COL32(0, 0, 139, 255), 4.0f);
	}

	for (const auto& i : disabled_intervals)
	{
		draw_list->AddLine(i.first, i.second, IM_COL32(139, 0, 0, 255), 4.0f);
	}
	ImGui::End();
	ImGui::PopStyleColor();
}

void View::startSimulation()
{
	model.setArrivalRate(arrival_rate);
	model.setBufferCapacity(buffer_capacity);
	model.setNumLines(num_lines);
	model.setReverseServiceTimeMean(reverse_service_time_mean);
	model.setSimulationTime(simulation_time);
	model.setFailureChance(failure_chance);
	model.setRecoveryRate(recovery_rate);
	model.startSimulation();
	if (isConcurency)
	{
		std::thread{ [this]() {
			while (model.getIsRunning())
			{
				std::lock_guard<std::mutex> lock(model_mutex);
				model.nextStep();
			}
			} }.detach();
	}
	initSimulationCanvas();
}

void View::updateUIData()
{
	std::lock_guard<std::mutex> lock(model_mutex);
	effectivity = model.getEfficiency();
	arrivals_count = model.getTotalArrivals();
	num_busy_lines = model.getNumBusyLines();
	buffer_usage = model.getCurrentBufferUsage();
	rejected_count = model.getRejectedCalls();
	is_model_running = model.getIsRunning();
	processed = model.getProcesedEvents();
	num_disabled_lines = model.getNumDisableLines();
	failures_count = model.getTotalFailures();
}

void View::updateIntervals()
{
	ImVec2 start, end;
	while (last_processed != processed.second)
	{
		QueueingModel::Event event = *last_processed;
		if (event.type == QueueingModel::Event::Types::SERVICE_START ||
			event.type == QueueingModel::Event::Types::RECOVERY_START)
		{
			line_start_times[event.line] = event.timeStamp;
		}
		else if (event.type == QueueingModel::Event::Types::SERVICE_END ||
			event.type == QueueingModel::Event::Types::RECOVERY_END)
		{
			float start_time = line_start_times[event.line];
			float end_time = event.timeStamp;
			int line = event.line;
			float scale = (lines[line].second.x - lines[line].first.x) / simulation_time;
			start.y = lines[line].first.y;
			start.x = lines[line].first.x + start_time * scale;
			end.y = lines[line].first.y;
			end.x = lines[line].first.x + end_time * scale;
			(event.type == QueueingModel::Event::Types::SERVICE_END ?
				busy_intervals : disabled_intervals).emplace_back(start, end);
		}
		++last_processed;
	}
}

void View::initLines()
{
	line_start_times.resize(num_lines);
	lines.resize(num_lines);
	for (size_t i = 0; i != lines.size(); ++i)
	{
		ImVec2 start;
		ImVec2 end;
		start.x = window_pos.x;
		end.x = window_pos.x + window_size.x;
		start.y = window_pos.y + (i + 1) * window_size.y / (num_lines + 1);
		end.y = start.y;
		lines[i] = { start, end };
	}
}

void View::initSimulationCanvas()
{
	initLines();
	busy_intervals.clear();
	disabled_intervals.clear();
	last_processed = model.getProcesedEvents().first;
}