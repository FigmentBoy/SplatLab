#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// A child process whose stdout and stderr are read line by line on a
// background thread. The child is terminated when this object is destroyed
// (on Windows it also lives in a kill-on-close job, so it never outlives the
// parent, even if the parent crashes).
class Subprocess {
public:
    // args[0] is the program (searched in PATH).
    static std::unique_ptr<Subprocess> create(const std::vector<std::string>& args);
    bool init(const std::vector<std::string>& args);
    ~Subprocess();

    Subprocess(const Subprocess&) = delete;
    Subprocess& operator=(const Subprocess&) = delete;

    // Output lines received since the last call (without line endings).
    std::vector<std::string> takeLines();

    // False once the process has exited and all its output has been read.
    bool running() const { return !m_finished.load(); }
    // Exit code, valid once !running().
    int exitCode() const { return m_exitCode.load(); }

    // Kills the process (and on POSIX its process group).
    void terminate();

private:
    Subprocess() = default;

    void readLoop();
    void pushData(const char* data, size_t size);

    std::thread m_reader;
    std::mutex m_mutex;
    std::vector<std::string> m_lines;
    std::string m_partial;
    std::atomic<bool> m_finished{false};
    std::atomic<int> m_exitCode{-1};

#ifdef _WIN32
    void* m_process = nullptr;
    void* m_job = nullptr;
    void* m_readPipe = nullptr;
#else
    int m_pid = -1;
    int m_readFd = -1;
#endif
};
