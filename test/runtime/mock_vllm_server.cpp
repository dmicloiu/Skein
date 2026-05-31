#include "mock_vllm_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace flock {
namespace {

std::string ToLower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string Trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// Read exactly n bytes (or until EOF/error). Returns true on success.
bool ReadExact(int fd, char* buf, size_t n) {
    size_t total = 0;
    while (total < n) {
        ssize_t r = ::read(fd, buf + total, n - total);
        if (r > 0) {
            total += static_cast<size_t>(r);
        } else if (r == 0) {
            return false;  // peer closed
        } else if (errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool WriteAll(int fd, const char* buf, size_t n) {
    size_t total = 0;
    while (total < n) {
        ssize_t w = ::write(fd, buf + total, n - total);
        if (w > 0) {
            total += static_cast<size_t>(w);
        } else if (w == -1 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

// Read until we see "\r\n\r\n". Stops appending after the marker; remaining
// already-buffered bytes are returned in *overflow.
bool ReadHeaders(int fd, std::string* headers_out, std::string* overflow) {
    headers_out->clear();
    overflow->clear();
    char ch;
    std::string buf;
    buf.reserve(1024);
    while (true) {
        ssize_t r = ::read(fd, &ch, 1);
        if (r == 1) {
            buf.push_back(ch);
            if (buf.size() >= 4 &&
                buf[buf.size() - 4] == '\r' && buf[buf.size() - 3] == '\n' &&
                buf[buf.size() - 2] == '\r' && buf[buf.size() - 1] == '\n') {
                *headers_out = buf;
                return true;
            }
            if (buf.size() > 64 * 1024) return false;  // too large
        } else if (r == 0) {
            return false;
        } else if (errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
}

bool ParseRequest(const std::string& head, MockVLLMServer::Request* req) {
    size_t pos = head.find("\r\n");
    if (pos == std::string::npos) return false;
    std::string request_line = head.substr(0, pos);
    // METHOD SP PATH SP HTTP/1.x
    size_t s1 = request_line.find(' ');
    if (s1 == std::string::npos) return false;
    size_t s2 = request_line.find(' ', s1 + 1);
    if (s2 == std::string::npos) return false;
    req->method = request_line.substr(0, s1);
    req->path = request_line.substr(s1 + 1, s2 - s1 - 1);

    size_t cursor = pos + 2;
    while (cursor < head.size()) {
        size_t eol = head.find("\r\n", cursor);
        if (eol == std::string::npos) break;
        if (eol == cursor) break;  // end of headers
        std::string line = head.substr(cursor, eol - cursor);
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string k = ToLower(Trim(line.substr(0, colon)));
            std::string v = Trim(line.substr(colon + 1));
            req->headers[k] = v;
        }
        cursor = eol + 2;
    }
    return true;
}

bool ShouldKeepAlive(const MockVLLMServer::Request& req) {
    auto it = req.headers.find("connection");
    if (it == req.headers.end()) return true;  // HTTP/1.1 default
    std::string v = ToLower(it->second);
    return v.find("close") == std::string::npos;
}

std::string BuildResponse(const MockVLLMServer::Response& resp, bool keep_alive) {
    static const char* kReason[] = {"OK", "Bad Request", "Internal Server Error"};
    const char* reason = "OK";
    if (resp.status == 400) reason = kReason[1];
    else if (resp.status >= 500) reason = kReason[2];
    else if (resp.status >= 400) reason = "Client Error";

    std::string out;
    out.reserve(128 + resp.body.size());
    out += "HTTP/1.1 ";
    out += std::to_string(resp.status);
    out += ' ';
    out += reason;
    out += "\r\n";
    out += "Content-Length: " + std::to_string(resp.body.size()) + "\r\n";
    out += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    for (const auto& kv : resp.headers) {
        out += kv.first + ": " + kv.second + "\r\n";
    }
    out += "\r\n";
    out += resp.body;
    return out;
}

}  // namespace

MockVLLMServer::MockVLLMServer() = default;

MockVLLMServer::~MockVLLMServer() { Stop(); }

std::string MockVLLMServer::Url(const std::string& path) const {
    std::string p = path;
    if (p.empty() || p[0] != '/') p = "/" + p;
    return "http://127.0.0.1:" + std::to_string(port_) + p;
}

void MockVLLMServer::Start(Handler handler) {
    handler_ = std::move(handler);

    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket() failed");

    int yes = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        throw std::runtime_error("bind() failed");
    }
    socklen_t len = sizeof(addr);
    if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        throw std::runtime_error("getsockname() failed");
    }
    port_ = ntohs(addr.sin_port);

    if (::listen(listen_fd_, 128) != 0) {
        throw std::runtime_error("listen() failed");
    }

    accept_thread_ = std::thread([this] { AcceptLoop(); });
}

void MockVLLMServer::Stop() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true)) return;

    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        // Defer the -1 assignment until after join: AcceptLoop still reads
        // listen_fd_ in its ::accept() call until the syscall returns from the
        // shutdown above.
    }
    if (accept_thread_.joinable()) accept_thread_.join();
    listen_fd_ = -1;

    // Join handler threads.
    std::vector<std::thread> ts;
    {
        std::lock_guard<std::mutex> lock(threads_mu_);
        ts.swap(handler_threads_);
    }
    for (auto& t : ts) {
        if (t.joinable()) t.join();
    }
}

void MockVLLMServer::AcceptLoop() {
    while (!stop_.load(std::memory_order_acquire)) {
        sockaddr_in client{};
        socklen_t clen = sizeof(client);
        int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client), &clen);
        if (fd < 0) {
            if (stop_.load(std::memory_order_acquire)) break;
            if (errno == EINTR) continue;
            break;
        }
        connections_.fetch_add(1, std::memory_order_relaxed);
        // Disable Nagle so test latency is predictable.
        int yes = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

        std::thread t([this, fd] { HandleConnection(fd); });
        std::lock_guard<std::mutex> lock(threads_mu_);
        handler_threads_.push_back(std::move(t));
    }
}

void MockVLLMServer::HandleConnection(int fd) {
    while (!stop_.load(std::memory_order_acquire)) {
        std::string head;
        std::string overflow;
        if (!ReadHeaders(fd, &head, &overflow)) break;

        Request req;
        if (!ParseRequest(head, &req)) break;

        size_t content_length = 0;
        auto it = req.headers.find("content-length");
        if (it != req.headers.end()) {
            try {
                content_length = static_cast<size_t>(std::stoul(it->second));
            } catch (...) {
                break;
            }
        }
        if (content_length > 0) {
            req.body.resize(content_length);
            if (!ReadExact(fd, req.body.data(), content_length)) break;
        }

        requests_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(recorded_mu_);
            recorded_.push_back(req);
        }

        Response resp = handler_(req);
        if (resp.delay.count() > 0) {
            std::this_thread::sleep_for(resp.delay);
        }
        if (stop_.load(std::memory_order_acquire)) break;

        bool keep = ShouldKeepAlive(req);
        std::string wire = BuildResponse(resp, keep);
        if (!WriteAll(fd, wire.data(), wire.size())) break;
        if (!keep) break;
    }
    ::close(fd);
}

std::vector<MockVLLMServer::Request> MockVLLMServer::RecordedRequests() const {
    std::lock_guard<std::mutex> lock(recorded_mu_);
    return recorded_;
}

}  // namespace flock
