#include "app/TrainingSession.hpp"

#include "io/Subprocess.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr size_t kMaxLogLines = 400;

fs::path executableDir() {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, buffer, static_cast<DWORD>(std::size(buffer)));
    return n ? fs::path(std::wstring(buffer, n)).parent_path() : fs::path();
#else
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path() : exe.parent_path();
#endif
}

// The repository root (the folder containing trainer/train.py), searched
// upwards from the working directory and from the executable.
fs::path findRepoRoot() {
    std::error_code ec;
    for (fs::path start : {fs::current_path(ec), executableDir()}) {
        for (fs::path dir = start; !dir.empty(); dir = dir.parent_path()) {
            if (fs::is_regular_file(dir / "trainer" / "train.py", ec)) return dir;
            if (dir == dir.parent_path()) break;
        }
    }
    return {};
}

bool hasModule(const fs::path& dir) {
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        std::string name = entry.path().filename().string();
        if (name.rfind("splat_vk", 0) == 0 &&
            (entry.path().extension() == ".pyd" || entry.path().extension() == ".so")) {
            return true;
        }
    }
    return false;
}

std::string getEnv(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t size = 0;
    std::string out;
    if (_dupenv_s(&value, &size, name) == 0 && value) out = value;
    free(value);
    return out;
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}

void setEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // namespace

std::string TrainingSession::locateTrainer(std::string& python, std::string& moduleDir) {
    fs::path root = findRepoRoot();
    if (root.empty()) return {};
    std::error_code ec;
    if (std::string env = getEnv("SPLAT_TRAINER_PYTHON"); !env.empty()) {
        python = env;
    } else if (fs::is_regular_file(root / "trainer" / ".venv" / "Scripts" / "python.exe", ec)) {
        python = (root / "trainer" / ".venv" / "Scripts" / "python.exe").string();
    } else if (fs::is_regular_file(root / "trainer" / ".venv" / "bin" / "python", ec)) {
        python = (root / "trainer" / ".venv" / "bin" / "python").string();
    } else {
#ifdef _WIN32
        python = "python";
#else
        python = "python3";
#endif
    }
    // Prefer the module built next to this executable (same build).
    fs::path exeDir = executableDir();
    moduleDir = hasModule(exeDir) ? exeDir.string() : std::string();
    return (root / "trainer" / "bootstrap.py").string();
}

std::unique_ptr<TrainingSession> TrainingSession::create(const TrainingRequest& request) {
    std::unique_ptr<TrainingSession> session(new TrainingSession());
    if (!session->init(request)) return nullptr;
    return session;
}

bool TrainingSession::init(const TrainingRequest& request) {
    m_request = request;
    std::string python, moduleDir;
    std::string script = locateTrainer(python, moduleDir);
    if (script.empty()) {
        std::cerr << "Training: trainer/train.py not found (run the viewer from the repository)\n";
        return false;
    }
    std::vector<std::string> args = {python, "-u", script, request.input, "-o", request.output,
                                      "--iterations", std::to_string(request.iterations),
                                      "--frames", std::to_string(request.frames),
                                      "--max-size", std::to_string(request.maxSize), "--progress-json"};
    if (!request.holdOutTestViews) args.push_back("--no-eval");
    if (!moduleDir.empty()) {
        args.push_back("--module-dir");
        args.push_back(moduleDir);
    }
    setEnv("PYTHONUNBUFFERED", "1");
    setEnv("PYTHONIOENCODING", "utf-8");

    std::string commandLine;
    for (const std::string& a : args) commandLine += (commandLine.empty() ? "" : " ") + a;
    std::cout << "Starting trainer: " << commandLine << "\n";
    appendLog("$ " + commandLine);
    m_process = Subprocess::create(args);
    if (!m_process) {
        m_status.failed = true;
        m_status.pythonMissing = true;
        m_status.error = "Python not installed";
        m_status.stageLabel = m_status.error;
        return true;
    }
    m_status.running = true;
    m_status.stageLabel = "Starting the trainer...";
    return true;
}

TrainingSession::~TrainingSession() = default;

void TrainingSession::cancel() {
    if (m_process && m_process->running()) {
        m_cancelled = true;
        m_process->terminate();
    }
}

void TrainingSession::appendLog(const std::string& line) {
    m_status.log.push_back(line);
    while (m_status.log.size() > kMaxLogLines) m_status.log.pop_front();
}

void TrainingSession::update(std::vector<Event>& events) {
    if (!m_process) return;
    for (const std::string& line : m_process->takeLines()) handleLine(line, events);
    if (m_status.running && !m_process->running()) {
        for (const std::string& line : m_process->takeLines()) handleLine(line, events);
        m_status.running = false;
        if (!m_status.succeeded) {
            m_status.failed = true;
            if (!m_sawHello && !m_cancelled) {
                // The interpreter never ran bootstrap.py: no Python (on Windows
                // the Store alias "python" exits with 9009), or a broken one.
                m_status.pythonMissing = true;
                m_status.error = "Python not installed";
            } else if (m_status.error.empty()) {
                m_status.error = m_cancelled ? "Cancelled"
                                             : "The trainer exited with code " + std::to_string(m_process->exitCode());
            }
        }
        m_status.stageLabel = m_status.succeeded ? "Finished" : m_status.error;
        appendLog(m_status.succeeded ? "Training finished." : "Training stopped: " + m_status.error);
    }
}

void TrainingSession::handleLine(const std::string& line, std::vector<Event>& events) {
    if (line.rfind("@@", 0) != 0) {
        std::cout << "[trainer] " << line << "\n";
        appendLog(line);
        return;
    }
    nlohmann::json e = nlohmann::json::parse(line.begin() + 2, line.end(), nullptr, false);
    if (e.is_discarded() || !e.is_object()) return;
    const std::string type = e.value("type", std::string());
    if (type == "hello") {
        m_sawHello = true;
    } else if (type == "stage") {
        m_status.stage = e.value("stage", std::string());
        m_status.stageLabel = e.value("label", std::string());
        m_status.progress = e.value("progress", -1.0f);
    } else if (type == "train") {
        m_status.iteration = e.value("iteration", 0);
        m_status.totalIterations = e.value("total", 0);
        m_status.gaussians = e.value("gaussians", uint64_t(0));
        if (e.contains("loss") && e["loss"].is_number()) m_status.loss = e["loss"].get<double>();
        if (e.contains("psnr") && e["psnr"].is_number()) m_status.psnr = e["psnr"].get<double>();
        if (e.contains("its") && e["its"].is_number()) m_status.itsPerSecond = e["its"].get<double>();
    } else if (type == "eval") {
        m_status.testPsnr = e.value("psnr", -1.0);
    } else if (type == "cameras") {
        events.push_back({Event::Kind::Cameras, e.value("path", std::string()), "Cameras"});
    } else if (type == "ply") {
        bool points = e.value("kind", std::string()) == "points";
        events.push_back({points ? Event::Kind::Points : Event::Kind::Gaussians, e.value("path", std::string()),
                          e.value("label", std::string())});
    } else if (type == "done") {
        m_status.succeeded = true;
        m_status.outputDir = e.value("output", m_status.outputDir);
        events.push_back({Event::Kind::Finished, m_status.outputDir, "Final model"});
    } else if (type == "error") {
        m_status.error = e.value("message", std::string("error"));
    }
}
