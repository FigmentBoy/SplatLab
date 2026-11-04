#include "gui/CamerasPanel.hpp"

#include "io/CameraSet.hpp"

#include <imgui.h>

#include <cstdio>
#include <filesystem>

std::unique_ptr<CamerasPanel> CamerasPanel::create(SelectCallback onSelect) {
    std::unique_ptr<CamerasPanel> panel(new CamerasPanel());
    if (!panel->init(std::move(onSelect))) return nullptr;
    return panel;
}

bool CamerasPanel::init(SelectCallback onSelect) {
    m_onSelect = std::move(onSelect);
    return static_cast<bool>(m_onSelect);
}

void CamerasPanel::setCameras(const CameraSet* cameras) {
    m_cameras = cameras;
    m_current = 0;
    m_active = false;
    m_scrollToCurrent = true;
}

void CamerasPanel::setCurrent(size_t index) {
    if (index != m_current || !m_active) m_scrollToCurrent = true;
    m_current = index;
    m_active = true;
}

void CamerasPanel::clearCurrent() {
    m_active = false;
}

void CamerasPanel::select(size_t index) {
    setCurrent(index);
    m_onSelect(index);
}

void CamerasPanel::drawMenuItems() {
    ImGui::MenuItem("Cameras", nullptr, &m_visible);
}

void CamerasPanel::drawWindows() {
    if (!m_visible) return;
    ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(280, 360), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Cameras", &m_visible)) {
        ImGui::End();
        return;
    }

    if (!m_cameras || m_cameras->empty()) {
        ImGui::TextDisabled("No cameras loaded.");
        ImGui::TextDisabled("Use File > Load Cameras or Load Scene.");
        ImGui::End();
        return;
    }

    size_t count = m_cameras->size();
    std::string file = std::filesystem::path(m_cameras->path()).filename().string();
    ImGui::TextDisabled("%s (%zu cameras)", file.c_str(), count);

    if (ImGui::ArrowButton("##prev", ImGuiDir_Left)) select((m_current + count - 1) % count);
    ImGui::SameLine();
    if (ImGui::ArrowButton("##next", ImGuiDir_Right)) select((m_current + 1) % count);
    ImGui::SameLine();
    const CameraRecord& cur = (*m_cameras)[m_current];
    if (m_active) {
        ImGui::Text("%zu / %zu  %s", m_current + 1, count, cur.imageName.c_str());
        ImGui::TextDisabled("%ux%u  fx %.1f  fy %.1f", cur.width, cur.height, cur.fx, cur.fy);
    } else {
        // Moved away from the dataset camera; the arrows step from the last one.
        ImGui::TextDisabled("Free camera (last: %zu)", m_current + 1);
        ImGui::TextDisabled("Pick a camera to snap back to it");
    }
    ImGui::Separator();

    if (ImGui::BeginChild("##list")) {
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(count));
        if (m_scrollToCurrent) clipper.IncludeItemByIndex(static_cast<int>(m_current));
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                const CameraRecord& rec = (*m_cameras)[static_cast<size_t>(i)];
                bool isCurrent = static_cast<size_t>(i) == m_current;
                ImGui::PushID(i);
                char label[160];
                std::snprintf(label, sizeof(label), "%3d  %s", i + 1, rec.imageName.c_str());
                if (ImGui::Selectable(label, m_active && isCurrent)) select(static_cast<size_t>(i));
                if (isCurrent && m_scrollToCurrent) {
                    ImGui::SetScrollHereY(0.5f);
                    m_scrollToCurrent = false;
                }
                ImGui::PopID();
            }
        }
    }
    ImGui::EndChild();
    ImGui::End();
}
