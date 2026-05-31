#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace flock {

// In-process HTTP/1.1 server for testing AsyncLLMClient.
//
// Binds to 127.0.0.1 on an ephemeral port; client tests connect to Url(path).
// Honors keep-alive (HTTP/1.1 default) so that connection-reuse via CURLSH
// can be observed via Connections() vs Requests().
class MockVLLMServer {
public:
    struct Request {
        std::string method;
        std::string path;
        // Header keys are lowercased for easy lookup.
        std::map<std::string, std::string> headers;
        std::string body;
    };

    struct Response {
        int status = 200;
        std::map<std::string, std::string> headers;
        std::string body;
        std::chrono::milliseconds delay{0};
    };

    using Handler = std::function<Response(const Request&)>;

    MockVLLMServer();
    ~MockVLLMServer();

    // Bind + listen + start the accept thread. Returns once the listening
    // port is known. handler is invoked on a worker thread per request.
    void Start(Handler handler);

    // Signal stop, shut down sockets, join all threads. Idempotent.
    void Stop();

    int Port() const { return port_; }
    std::string Url(const std::string& path) const;

    // Total TCP connections accepted (live + closed).
    int Connections() const { return connections_.load(); }
    // Total HTTP requests fully received.
    int Requests() const { return requests_.load(); }

    // Snapshot of all received requests (in arrival order).
    std::vector<Request> RecordedRequests() const;

private:
    void AcceptLoop();
    void HandleConnection(int fd);

    int listen_fd_ = -1;
    int port_ = 0;
    Handler handler_;

    std::atomic<bool> stop_{false};
    std::atomic<int> connections_{0};
    std::atomic<int> requests_{0};

    std::thread accept_thread_;
    std::mutex threads_mu_;
    std::vector<std::thread> handler_threads_;

    mutable std::mutex recorded_mu_;
    std::vector<Request> recorded_;
};

}  // namespace flock
