#include "gui/TrainPanel.hpp"

#include "io/ModelLibrary.hpp"

#include <imgui.h>
#include <nfd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

void copyString(char* dst, size_t size, const std::string& src) {
    std::snprintf(dst, size, "%s", src.c_str());
}

std::string formatDuration(double seconds) {
    if (seconds < 0 || seconds > 1e7) return "?";
    int s = static_cast<int>(seconds + 0.5);
    char buf[32];
    if (s >= 3600) std::snprintf(buf, sizeof(buf), "%dh %02dm", s / 3600, (s / 60) % 60);
    else std::snprintf(buf, sizeof(buf), "%dm %02ds", s / 60, s % 60);
    return buf;
}

} // namespace

std::unique_ptr<TrainPanel> TrainPanel::create(StartCallback onStart, CancelCallback onCancel) {
    std::unique_ptr<TrainPanel> panel(new TrainPanel());
    if (!panel->init(std::move(onStart), std::move(onCancel))) return nullptr;
    return panel;
}

bool TrainPanel::init(StartCallback onStart, CancelCallback onCancel) {
    m_onStart = std::move(onStart);
    m_onCancel = std::move(onCancel);
    m_nfdInitialized = NFD_Init() == NFD_OKAY;
    std::unique_ptr<ModelLibrary> library = ModelLibrary::create();
    if (library && !library->rootPath().empty()) {
        m_modelsRoot = library->rootPath();
    } else {
        std::error_code ec;
        m_modelsRoot = (fs::current_path(ec) / "data" / "models").string();
    }
    return m_onStart && m_onCancel;
}

TrainPanel::~TrainPanel() {
    if (m_nfdInitialized) NFD_Quit();
}

void TrainPanel::drawMenuItems() {
    if (ImGui::MenuItem("Train from Video...", nullptr, m_visible)) m_visible = !m_visible;
}

void TrainPanel::suggestOutput() {
    if (m_outputEdited || m_input[0] == '\0') return;
    fs::path input = fs::u8path(m_input);
    std::string stem = input.has_extension() ? input.stem().u8string() : input.filename().u8string();
    if (stem.empty()) stem = "trained";
    fs::path out = fs::u8path(m_modelsRoot) / fs::u8path(stem);
    // Do not overwrite an existing model by default.
    std::error_code ec;
    for (int i = 2; fs::exists(out / "point_cloud", ec) && i < 100; ++i) {
        out = fs::u8path(m_modelsRoot) / fs::u8path(stem + "_" + std::to_string(i));
    }
    copyString(m_output, sizeof(m_output), out.u8string());
}

void TrainPanel::drawWindows() {
    if (m_status && m_status->pythonMissing && !m_pythonDismissed) drawPythonMissing();
    if (!m_visible) return;
    ImGui::SetNextWindowSize(ImVec2(560.0f, 0.0f), ImGuiCond_FirstUseEver);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 580.0f, viewport->WorkPos.y + 30.0f),
                            ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Train from Video", &m_visible)) {
        bool running = m_status && m_status->running;
        if (running) m_showSetup = false;
        if (m_showSetup || !m_status) drawSetup();
        else drawRun(*m_status);
    }
    ImGui::End();
}

void TrainPanel::drawSetup() {
    ImGui::TextWrapped("Reconstructs a 3D Gaussian splat from a video: frames are extracted, camera poses "
                       "are estimated with COLMAP, and the Gaussians are optimized with the Vulkan "
                       "renderer's backward pass. Progress is shown in the viewer as it trains.");
    ImGui::Separator();

    ImGui::TextUnformatted("Video (or a folder of images)");
    ImGui::SetNextItemWidth(-90.0f);
    if (ImGui::InputText("##input", m_input, sizeof(m_input))) suggestOutput();
    ImGui::SameLine();
    if (ImGui::Button("Browse...##video") && m_nfdInitialized) {
        nfdu8filteritem_t filters[] = {{"Video", "mp4,mov,avi,mkv,webm,m4v,wmv"}};
        nfdu8char_t* path = nullptr;
        if (NFD_OpenDialogU8(&path, filters, 1, nullptr) == NFD_OKAY) {
            copyString(m_input, sizeof(m_input), path);
            NFD_FreePathU8(path);
            suggestOutput();
        }
    }

    ImGui::TextUnformatted("Output model folder");
    ImGui::SetNextItemWidth(-90.0f);
    if (ImGui::InputText("##output", m_output, sizeof(m_output))) m_outputEdited = true;
    ImGui::SameLine();
    if (ImGui::Button("Browse...##out") && m_nfdInitialized) {
        nfdu8char_t* path = nullptr;
        if (NFD_PickFolderU8(&path, nullptr) == NFD_OKAY) {
            copyString(m_output, sizeof(m_output), path);
            NFD_FreePathU8(path);
            m_outputEdited = true;
        }
    }

    ImGui::Spacing();
    ImGui::SetNextItemWidth(160.0f);
    ImGui::InputInt("Iterations", &m_iterations, 1000, 5000);
    m_iterations = std::clamp(m_iterations, 100, 200000);
    ImGui::SetNextItemWidth(160.0f);
    ImGui::InputInt("Frames to extract", &m_frames, 10, 50);
    m_frames = std::clamp(m_frames, 10, 2000);
    ImGui::SetNextItemWidth(160.0f);
    ImGui::InputInt("Max image size (px)", &m_maxSize, 100, 400);
    m_maxSize = std::clamp(m_maxSize, 320, 4096);
    ImGui::Checkbox("Hold out every 8th frame for evaluation", &m_holdOut);

    ImGui::Spacing();
    bool ready = m_input[0] != '\0' && m_output[0] != '\0';
    std::error_code ec;
    bool inputExists = ready && fs::exists(fs::u8path(m_input), ec);
    if (ready && !inputExists) ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.4f, 1.0f), "The input does not exist.");
    ImGui::BeginDisabled(!inputExists);
    if (ImGui::Button("Start training", ImVec2(160.0f, 0.0f))) {
        TrainingRequest request;
        request.input = m_input;
        request.output = m_output;
        request.iterations = m_iterations;
        request.frames = m_frames;
        request.maxSize = m_maxSize;
        request.holdOutTestViews = m_holdOut;
        m_showSetup = false;
        m_pythonDismissed = false;
        m_onStart(request);
    }
    ImGui::EndDisabled();
    if (m_status && !m_status->running) {
        ImGui::SameLine();
        if (ImGui::Button("Show last run")) m_showSetup = false;
    }
}

void TrainPanel::drawRun(const TrainingStatus& st) {
    ImGui::TextWrapped("Output: %s", st.outputDir.c_str());
    ImGui::Separator();

    if (st.running) {
        ImGui::TextUnformatted(st.stageLabel.c_str());
        if (st.progress < 0.0f) {
            ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-FLT_MIN, 0.0f), "");
        } else {
            char label[32];
            std::snprintf(label, sizeof(label), "%.0f%%", st.progress * 100.0f);
            ImGui::ProgressBar(st.progress, ImVec2(-FLT_MIN, 0.0f), label);
        }
    } else if (st.succeeded) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Training finished.");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.4f, 1.0f), "Training stopped: %s", st.error.c_str());
    }

    if (st.totalIterations > 0) {
        ImGui::Text("Iteration %d / %d", st.iteration, st.totalIterations);
        ImGui::Text("Loss %.4f   PSNR %.2f dB   %llu Gaussians", st.loss, st.psnr,
                    static_cast<unsigned long long>(st.gaussians));
        if (st.running && st.itsPerSecond > 0.0) {
            ImGui::Text("%.1f it/s, about %s left", st.itsPerSecond,
                        formatDuration((st.totalIterations - st.iteration) / st.itsPerSecond).c_str());
        }
        if (st.testPsnr > 0.0) ImGui::Text("Held-out views: PSNR %.2f dB", st.testPsnr);
    }
    if (!st.snapshotLabel.empty()) ImGui::TextDisabled("Showing: %s", st.snapshotLabel.c_str());

    if (ImGui::CollapsingHeader("Log", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginChild("##trainlog", ImVec2(0.0f, 220.0f), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_HorizontalScrollbar);
        for (const std::string& line : st.log) ImGui::TextUnformatted(line.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20.0f) ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();
    }

    if (st.running) {
        if (ImGui::Button("Cancel training")) m_onCancel();
    } else if (ImGui::Button("New training...")) {
        m_showSetup = true;
    }
}

void TrainPanel::drawPythonMissing() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize;
    if (ImGui::Begin("##pythonmissing", nullptr, flags)) {
        ImGui::SetWindowFontScale(1.6f);
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.45f, 1.0f), "Python not installed");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::TextUnformatted("Training needs Python 3.9+ (python.org). Install it, then start training again.");
        ImGui::TextDisabled("Or set SPLAT_TRAINER_PYTHON to the interpreter to use.");
        if (ImGui::Button("OK", ImVec2(120.0f, 0.0f))) {
            m_pythonDismissed = true;
            m_showSetup = true;
        }
    }
    ImGui::End();
}
