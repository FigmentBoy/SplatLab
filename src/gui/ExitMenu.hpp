#pragma once

#include "gui/MenuModule.hpp"

#include <climits>
#include <functional>
#include <memory>

// "File > Exit", always the last entry of the File menu.
class ExitMenu : public MenuModule {
public:
    using ExitCallback = std::function<void()>;

    static std::unique_ptr<ExitMenu> create(ExitCallback onExit);
    bool init(ExitCallback onExit);

    const char* menuName() const override { return "File"; }
    int menuOrder() const override { return INT_MAX; }
    void drawMenuItems() override;

private:
    ExitMenu() = default;

    ExitCallback m_onExit;
};
