#pragma once
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include "QueueingModel.hpp"
#include <vector>
#include <mutex>

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

	void drawRightPanel();

	void drawSystemVisualisation();

	void render();

	void startSimulation();

	void updateUIData();

	void updateIntervals();

	void initLines();

	void initSimulationCanvas();

private:
	// UI Utilities
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
	// ImVec4 clear_color = ImVec4(0.96f, 0.90f, 0.80f, 1.00f);
	bool done;

	// Setup window
	bool show_right_window = true;
	std::string lastError;
	bool is_model_running = false;
	bool show_error_popup = false;

	// Input
	float simulation_time = 100.0f;
	int num_lines = 1;
	int buffer_capacity = 0;
	float arrival_rate = 0.1f;
	float reverse_service_time_mean = 0.1f;
	bool isConcurency = false;

	// Output
	float effectivity = 1.f;
	int arrivals_count = 1;
	int num_busy_lines = 0;
	int buffer_usage = 0;
	int rejected_count = 0;
	std::mutex model_mutex;

	// Animation
	bool show_animation_window = false;
	ImVec2 window_pos = ImVec2(0, 0);
	ImVec2 window_size = ImVec2(880, 720);
	ImVec2 window_center = ImVec2(window_size.x / 2, window_size.y / 2);
	std::vector<std::pair<ImVec2, ImVec2>> lines;
	std::vector<std::pair<ImVec2, ImVec2>> intervals;
	std::vector<float> line_start_times;
	std::multiset<QueueingModel::Event>::const_iterator last_processed;
	std::pair<std::multiset<QueueingModel::Event>::const_iterator, std::multiset<QueueingModel::Event>::const_iterator>
		processed;

	// Model
	QueueingModel model;
};