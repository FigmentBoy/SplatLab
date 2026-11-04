#pragma once

#include "gui/MenuModule.hpp"

#include <functional>
#include <memory>

class ModelLibrary;
struct ModelEntry;

// "File > Load Scene": a submenu with every trained model under data/models.
// The list is re-read each time the File menu opens.
class ModelMenu : public MenuModule {
public:
    using LoadModelCallback = std::function<void(const ModelEntry& model)>;

    static std::unique_ptr<ModelMenu> create(LoadModelCallback onLoadModel);
    bool init(LoadModelCallback onLoadModel);
    ~ModelMenu() override;

    const char* menuName() const override { return "File"; }
    int menuOrder() const override { return -10; } // first entry of the File menu
    void drawMenuItems() override;

    // Disabled while training.
    void setEnabled(bool enabled) { m_enabled = enabled; }

private:
    ModelMenu() = default;

    LoadModelCallback m_onLoadModel;
    bool m_enabled = true;
    std::unique_ptr<ModelLibrary> m_library;
};
