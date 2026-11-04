#include "io/Subprocess.hpp"

#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// Quotes one argument following the MSVC runtime's command-line parsing rules.
std::wstring quoteArg(const std::wstring& arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) return arg;
    std::wstring out = L"\"";
    for (size_t i = 0;; ++i) {
        size_t backslashes = 0;
        while (i < arg.size() && arg[i] == L'\\') {
            ++i;
            ++backslashes;
        }
        if (i == arg.size()) {
            out.append(backslashes * 2, L'\\');
            break;
        }
        if (arg[i] == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
        } else {
            out.append(backslashes, L'\\');
        }
        out.push_back(arg[i]);
    }
    out.push_back(L'"');
    return out;
}
#endif

} // namespace

std::unique_ptr<Subprocess> Subprocess::create(const std::vector<std::string>& args) {
    std::unique_ptr<Subprocess> process(new Subprocess());
    if (!process->init(args)) return nullptr;
    return process;
}

#ifdef _WIN32

bool Subprocess::init(const std::vector<std::string>& args) {
    if (args.empty()) return false;
    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE readPipe = nullptr, writePipe = nullptr;
    if (!CreatePipe(&readPipe, &writePipe, &sa, 0)) return false;
    SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);

    std::wstring cmd;
    for (const std::string& a : args) {
        if (!cmd.empty()) cmd += L' ';
        cmd += quoteArg(widen(a));
    }

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = writePipe;
    si.hStdError = writePipe;
    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED,
                             nullptr, nullptr, &si, &pi);
    CloseHandle(writePipe);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        std::cerr << "Cannot start '" << args[0] << "' (error " << GetLastError() << ")\n";
        CloseHandle(readPipe);
        return false;
    }

    // Kill the child together with this process.
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
        AssignProcessToJobObject(job, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    m_process = pi.hProcess;
    m_job = job;
    m_readPipe = readPipe;
    m_reader = std::thread([this] { readLoop(); });
    return true;
}

void Subprocess::readLoop() {
    char buffer[4096];
    DWORD got = 0;
    while (ReadFile(static_cast<HANDLE>(m_readPipe), buffer, sizeof(buffer), &got, nullptr) && got > 0) {
        pushData(buffer, got);
    }
    WaitForSingleObject(static_cast<HANDLE>(m_process), INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(static_cast<HANDLE>(m_process), &code);
    pushData("\n", 1);
    m_exitCode = static_cast<int>(code);
    m_finished = true;
}

void Subprocess::terminate() {
    if (m_process && !m_finished) TerminateProcess(static_cast<HANDLE>(m_process), 1);
}

Subprocess::~Subprocess() {
    terminate();
    if (m_reader.joinable()) m_reader.join();
    if (m_readPipe) CloseHandle(static_cast<HANDLE>(m_readPipe));
    if (m_process) CloseHandle(static_cast<HANDLE>(m_process));
    if (m_job) CloseHandle(static_cast<HANDLE>(m_job));
}

#else

bool Subprocess::init(const std::vector<std::string>& args) {
    if (args.empty()) return false;
    int fds[2];
    if (pipe(fds) != 0) return false;
    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        setpgid(0, 0);
        int nul = open("/dev/null", O_RDONLY);
        if (nul >= 0) dup2(nul, 0);
        dup2(fds[1], 1);
        dup2(fds[1], 2);
        close(fds[0]);
        close(fds[1]);
        std::vector<char*> argv;
        for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    close(fds[1]);
    m_pid = pid;
    m_readFd = fds[0];
    m_reader = std::thread([this] { readLoop(); });
    return true;
}

void Subprocess::readLoop() {
    char buffer[4096];
    for (;;) {
        ssize_t got = read(m_readFd, buffer, sizeof(buffer));
        if (got <= 0) break;
        pushData(buffer, static_cast<size_t>(got));
    }
    int status = 0;
    waitpid(m_pid, &status, 0);
    pushData("\n", 1);
    m_exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    m_finished = true;
}

void Subprocess::terminate() {
    if (m_pid > 0 && !m_finished) kill(-m_pid, SIGTERM);
}

Subprocess::~Subprocess() {
    terminate();
    if (m_reader.joinable()) m_reader.join();
    if (m_readFd >= 0) close(m_readFd);
}

#endif

void Subprocess::pushData(const char* data, size_t size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (size_t i = 0; i < size; ++i) {
        char c = data[i];
        if (c == '\n') {
            if (!m_partial.empty() && m_partial.back() == '\r') m_partial.pop_back();
            if (!m_partial.empty()) m_lines.push_back(std::move(m_partial));
            m_partial.clear();
        } else {
            m_partial.push_back(c);
        }
    }
}

std::vector<std::string> Subprocess::takeLines() {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<std::string> lines;
    lines.swap(m_lines);
    return lines;
}
