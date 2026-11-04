#pragma once

#include <string>

// A pluggable piece of the main menu bar. Each module names the top-level menu
// it belongs to; modules that share a name are drawn into the same menu,
// sorted by menuOrder() (ties keep the order they were added to the Gui).
// Modules may also draw their own windows every frame.
//
// To add a feature to the UI, derive from MenuModule (following the
// create/init pattern) and register it with Gui::addModule().
class MenuModule {
public:
    virtual ~MenuModule() = default;

    MenuModule(const MenuModule&) = delete;
    MenuModule& operator=(const MenuModule&) = delete;

    // Top-level menu this module lives in, e.g. "File".
    virtual const char* menuName() const = 0;

    // Position within the menu: lower values are drawn first.
    virtual int menuOrder() const { return 0; }

    // Draws the module's items inside its menu (called between BeginMenu/EndMenu).
    virtual void drawMenuItems() = 0;

    // Draws any windows the module owns (called once per frame).
    virtual void drawWindows() {}

protected:
    // Protected rather than private so concrete modules can construct their base.
    MenuModule() = default;
};
