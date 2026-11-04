#include "gui/ExitMenu.hpp"

#include <imgui.h>

std::unique_ptr<ExitMenu> ExitMenu::create(ExitCallback onExit) {
    std::unique_ptr<ExitMenu> menu(new ExitMenu());
    if (!menu->init(std::move(onExit))) return nullptr;
    return menu;
}

bool ExitMenu::init(ExitCallback onExit) {
    m_onExit = std::move(onExit);
    return static_cast<bool>(m_onExit);
}

void ExitMenu::drawMenuItems() {
    ImGui::Separator();
    if (ImGui::MenuItem("Exit")) m_onExit();
}
