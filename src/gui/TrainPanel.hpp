#pragma once

#include "app/TrainingSession.hpp"
#include "gui/MenuModule.hpp"

#include <functional>
#include <memory>
#include <string>

// "Train > Train from Video...": a window to pick a video and start the
// trainer, which then shows the run's progress (stage, progress bar,
// iteration statistics, log) with a Cancel button. The viewer owns the
// TrainingSession and passes its status in every frame.
class TrainPanel : public MenuModule {
public:
    using StartCallback = std::function<void(const TrainingRequest& request)>;
    using CancelCallback = std::function<void()>;

    static std::unique_ptr<TrainPanel> create(StartCallback onStart, CancelCallback onCancel);
    bool init(StartCallback onStart, CancelCallback onCancel);
    ~TrainPanel() override;

    const char* menuName() const override { return "Train"; }
    void drawMenuItems() override;
    void drawWindows() override;

    // Status of the current (or last) run; nullptr if there was none.
    void setStatus(const TrainingStatus* status) { m_status = status; }
    void setVisible(bool visible) { m_visible = visible; }
    bool visible() const { return m_visible; }
    // Folder new models go into by default (data/models).
    const std::string& modelsRoot() const { return m_modelsRoot; }

private:
    TrainPanel() = default;

    void drawSetup();
    void drawRun(const TrainingStatus& status);
    void drawPythonMissing();
    void suggestOutput();

    StartCallback m_onStart;
    CancelCallback m_onCancel;
    const TrainingStatus* m_status = nullptr;
    bool m_visible = false;
    bool m_nfdInitialized = false;
    bool m_outputEdited = false;
    bool m_showSetup = true; // setup form instead of the last run's status
    bool m_pythonDismissed = false;

    char m_input[1024] = {};
    char m_output[1024] = {};
    int m_iterations = 30000;
    int m_frames = 200;
    int m_maxSize = 1600;
    bool m_holdOut = true;
    std::string m_modelsRoot;
};
