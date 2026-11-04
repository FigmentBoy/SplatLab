#include "gui/FileMenu.hpp"

#include <imgui.h>
#include <nfd.h>

#include <iostream>

std::unique_ptr<FileMenu> FileMenu::create(LoadPlyCallback onLoadPly, LoadCamerasCallback onLoadCameras) {
    std::unique_ptr<FileMenu> menu(new FileMenu());
    if (!menu->init(std::move(onLoadPly), std::move(onLoadCameras))) return nullptr;
    return menu;
}

bool FileMenu::init(LoadPlyCallback onLoadPly, LoadCamerasCallback onLoadCameras) {
    m_onLoadPly = std::move(onLoadPly);
    m_onLoadCameras = std::move(onLoadCameras);
    if (NFD_Init() != NFD_OKAY) {
        std::cerr << "Native file dialog init failed: " << NFD_GetError() << "\n";
        return false;
    }
    m_nfdInitialized = true;
    return true;
}

FileMenu::~FileMenu() {
    if (m_nfdInitialized) NFD_Quit();
}

void FileMenu::drawMenuItems() {
    std::string path;
    if (ImGui::MenuItem("Load PLY...", nullptr, false, m_enabled) && openDialog("Gaussian splat PLY", "ply", path) && m_onLoadPly) {
        m_onLoadPly(path);
    }
    if (ImGui::MenuItem("Load Cameras...", nullptr, false, m_enabled) && openDialog("3DGS cameras.json", "json", path) && m_onLoadCameras) {
        m_onLoadCameras(path);
    }
}

bool FileMenu::openDialog(const char* filterName, const char* filterSpec, std::string& path) {
    nfdu8filteritem_t filters[] = {{filterName, filterSpec}};
    nfdu8char_t* selected = nullptr;
    nfdresult_t result = NFD_OpenDialogU8(&selected, filters, 1, nullptr);
    if (result == NFD_OKAY) {
        path = selected;
        NFD_FreePathU8(selected);
        return true;
    }
    if (result == NFD_ERROR) std::cerr << "File dialog error: " << NFD_GetError() << "\n";
    return false;
}
