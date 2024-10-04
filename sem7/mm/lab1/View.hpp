#pragma once
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>

class View
{
public:
	View();

	~View();

	void draw();

	bool isDone() const;

private:
	void processEvent();

	void newFrame();

	void drawWidgets();

	void render();

private:
	SDL_WindowFlags window_flags;
	SDL_Window* window;
	ImVector<const char*> extensions;
	uint32_t extensions_count;
	VkSurfaceKHR surface;
	VkResult err;
	int w, h;
	ImGui_ImplVulkanH_Window* wd;
	ImGuiIO* io;
	ImGui_ImplVulkan_InitInfo init_info;
	bool show_demo_window;
	bool show_another_window;
	ImVec4 clear_color;
	bool done;
};