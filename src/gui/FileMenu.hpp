#pragma once

#include "gui/MenuModule.hpp"

#include <functional>
#include <memory>
#include <string>

// "File" menu: loading a PLY or a cameras.json through the native file dialog.
class FileMenu : public MenuModule {
public:
    using LoadPlyCallback = std::function<void(const std::string& path)>;
    using LoadCamerasCallback = std::function<void(const std::string& path)>;

    static std::unique_ptr<FileMenu> create(LoadPlyCallback onLoadPly, LoadCamerasCallback onLoadCameras);
    bool init(LoadPlyCallback onLoadPly, LoadCamerasCallback onLoadCameras);
    ~FileMenu() override;

    const char* menuName() const override { return "File"; }
    void drawMenuItems() override;

    // Disabled while training (the viewer then shows the trainer's progress).
    void setEnabled(bool enabled) { m_enabled = enabled; }

private:
    FileMenu() = default;

    // Shows an open-file dialog; returns false if it was cancelled or failed.
    static bool openDialog(const char* filterName, const char* filterSpec, std::string& path);

    LoadPlyCallback m_onLoadPly;
    LoadCamerasCallback m_onLoadCameras;
    bool m_nfdInitialized = false;
    bool m_enabled = true;
};
