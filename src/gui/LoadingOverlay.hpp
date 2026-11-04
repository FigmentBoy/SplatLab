#pragma once

#include <memory>
#include <string>

// A small window centered on the screen with a status line and a progress
// bar, drawn while something is loading. Call draw() between
// Gui::beginFrame() and Gui::endFrame().
class LoadingOverlay {
public:
    static std::unique_ptr<LoadingOverlay> create();
    bool init();

    // `fraction` in 0..1; a negative value shows an indeterminate bar.
    void draw(const std::string& status, float fraction);

private:
    LoadingOverlay() = default;
};
