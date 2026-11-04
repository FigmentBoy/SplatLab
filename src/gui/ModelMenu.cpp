#include "gui/ModelMenu.hpp"

#include "io/ModelLibrary.hpp"

#include <imgui.h>

std::unique_ptr<ModelMenu> ModelMenu::create(LoadModelCallback onLoadModel) {
    std::unique_ptr<ModelMenu> menu(new ModelMenu());
    if (!menu->init(std::move(onLoadModel))) return nullptr;
    return menu;
}

bool ModelMenu::init(LoadModelCallback onLoadModel) {
    m_onLoadModel = std::move(onLoadModel);
    m_library = ModelLibrary::create();
    return m_onLoadModel && m_library;
}

ModelMenu::~ModelMenu() = default;

void ModelMenu::drawMenuItems() {
    // Called inside the File menu's popup window: rescan whenever it opens.
    if (ImGui::IsWindowAppearing()) m_library->refresh();

    if (!ImGui::BeginMenu("Load Scene", m_enabled)) return;
    const auto& models = m_library->models();
    if (m_library->rootPath().empty()) {
        ImGui::TextDisabled("No data/models folder found");
    } else if (models.empty()) {
        ImGui::TextDisabled("No models in %s", m_library->rootPath().c_str());
    }
    for (const ModelEntry& model : models) {
        if (ImGui::MenuItem(model.name.c_str())) m_onLoadModel(model);
        if (model.camerasPath.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("No cameras.json");
    }
    ImGui::EndMenu();
}
