#pragma once

#include "gui/MenuModule.hpp"

#include <functional>
#include <memory>

class CameraSet;

// "View > Cameras": a window listing the loaded dataset cameras, with
// previous/next buttons. Selecting one calls back into the viewer.
class CamerasPanel : public MenuModule {
public:
    using SelectCallback = std::function<void(size_t index)>;

    static std::unique_ptr<CamerasPanel> create(SelectCallback onSelect);
    bool init(SelectCallback onSelect);

    const char* menuName() const override { return "View"; }
    void drawMenuItems() override;
    void drawWindows() override;

    // The panel does not own the set; pass nullptr when it goes away.
    void setCameras(const CameraSet* cameras);
    // Highlights `index` (e.g. after the viewer switched cameras by hotkey).
    void setCurrent(size_t index);
    // Removes the highlight once the view no longer matches a dataset camera.
    void clearCurrent();
    void setVisible(bool visible) { m_visible = visible; }

private:
    CamerasPanel() = default;

    void select(size_t index);

    SelectCallback m_onSelect;
    const CameraSet* m_cameras = nullptr;
    size_t m_current = 0;  // active camera, or the last one while !m_active
    bool m_active = false; // the view is exactly camera m_current
    bool m_visible = false;
    bool m_scrollToCurrent = false;
};
