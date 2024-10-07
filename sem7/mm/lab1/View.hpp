#pragma once
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include "PlanetSystem.hpp"
#include <vector>

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

	void drawSystemSetup();

	void drawSystemState();

	void drawFileActions();

	void drawSystemVisualisation();

	void render();

	void startSimulation();

	void updateUIData();

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
	bool show_demo_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	bool done;

	// Setup window
	bool show_right_window = true;
	float time_step = 3600.0f;
	float simulation_time = 360000000.0f;
	int selected_scheme = 0;
	int num_planets = 0;
	std::vector<ImVec2> positions;
	std::vector<ImVec2> velocities;
	std::vector<float> masses;

	// State window
	bool show_left_window = true;
	float total_energy = 0.0f;
	float current_time = 0.0f;
	ImVec2 center_of_mass_velocity = ImVec2{ 0.0f, 0.0f };
	bool is_model_running = false;

	// File window
	bool show_left_bottom_window = true;

	// Animation
	bool show_animation_window = false;
	ImVec2 window_pos = ImVec2(0, 0);
	ImVec2 window_size = ImVec2(880, 720);
	float scale = 1.0f;
	float angle = 0.0f;
	ImVec2 center = ImVec2(window_size.x / 2, window_size.y / 2);
	ImVec2 point_pos = center;
	ImVector<ImVec2> points;

	// Model
	PlanetSystem system;
	size_t lastStep = 0;
};