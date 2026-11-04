#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

class Subprocess;

// What to train (the settings of the Train window).
struct TrainingRequest {
    std::string input;  // video file or image folder
    std::string output; // model folder to create
    int iterations = 30000;
    int frames = 200;
    int maxSize = 1600;
    bool holdOutTestViews = true;
};

// Progress of a run, as reported by the trainer's JSON events.
struct TrainingStatus {
    bool running = false;
    bool succeeded = false;
    bool failed = false;
    std::string stage;      // stage id, e.g. "mapping", "densify"
    std::string stageLabel; // human-readable stage
    float progress = -1.0f; // 0..1, negative if unknown
    int iteration = 0;
    int totalIterations = 0;
    double loss = 0.0;
    double psnr = 0.0;
    double itsPerSecond = 0.0;
    uint64_t gaussians = 0;
    double testPsnr = -1.0;
    std::string snapshotLabel; // what the viewer currently shows
    std::string error;
    bool pythonMissing = false; // Python could not be started at all
    std::string outputDir;
    std::deque<std::string> log; // recent plain output lines
};

// Runs the Python trainer (trainer/bootstrap.py, which installs missing
// packages and then runs train.py) as a child process, and turns its
// "@@{json}" progress lines into status updates and scene-load events.
class TrainingSession {
public:
    struct Event {
        enum class Kind { Cameras, Points, Gaussians, Finished };
        Kind kind;
        std::string path; // cameras.json, a PLY, or (Finished) the output folder
        std::string label;
    };

    // Always returns a session; if the trainer cannot be started its status
    // is failed (with pythonMissing set when Python itself is missing).
    static std::unique_ptr<TrainingSession> create(const TrainingRequest& request);
    bool init(const TrainingRequest& request);
    ~TrainingSession();

    TrainingSession(const TrainingSession&) = delete;
    TrainingSession& operator=(const TrainingSession&) = delete;

    // Reads the trainer's output; appends scene events to `events`.
    void update(std::vector<Event>& events);
    void cancel();
    // What the viewer currently shows (displayed in the Train window).
    void setShowing(const std::string& label) { m_status.snapshotLabel = label; }

    const TrainingStatus& status() const { return m_status; }
    const TrainingRequest& request() const { return m_request; }

    // Where the trainer was found (for error messages in the UI).
    static std::string locateTrainer(std::string& python, std::string& moduleDir);

private:
    TrainingSession() = default;

    void handleLine(const std::string& line, std::vector<Event>& events);
    void appendLog(const std::string& line);

    TrainingRequest m_request;
    TrainingStatus m_status;
    std::unique_ptr<Subprocess> m_process;
    bool m_cancelled = false;
    bool m_sawHello = false; // bootstrap.py started, so Python works
};
