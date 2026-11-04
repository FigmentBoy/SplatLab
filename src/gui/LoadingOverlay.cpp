#include "gui/LoadingOverlay.hpp"

#include <imgui.h>

#include <cstdio>

std::unique_ptr<LoadingOverlay> LoadingOverlay::create() {
    std::unique_ptr<LoadingOverlay> overlay(new LoadingOverlay());
    if (!overlay->init()) return nullptr;
    return overlay;
}

bool LoadingOverlay::init() {
    return true;
}

void LoadingOverlay::draw(const std::string& status, float fraction) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowFocus(); // stay above the other windows
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("##loading", nullptr, flags)) {
        ImGui::TextUnformatted(status.c_str());
        if (fraction < 0.0f) {
            // Negative fractions animate an indeterminate bar.
            ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-FLT_MIN, 0.0f), "");
        } else {
            char label[16];
            std::snprintf(label, sizeof(label), "%.0f%%", fraction * 100.0f);
            ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), label);
        }
    }
    ImGui::End();
}
