#include "flock/runtime/async_llm_client.h"
#include "mock_vllm_server.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace flock {
namespace {

using namespace std::chrono_literals;

// Helper: wait until predicate is true, or fail after timeout.
template <class Pred>
bool WaitFor(std::chrono::milliseconds timeout, Pred pred) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(2ms);
    }
    return pred();
}

class AsyncLLMClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        server_ = std::make_unique<MockVLLMServer>();
    }
    void TearDown() override {
        client_.reset();
        if (server_) server_->Stop();
        server_.reset();
    }

    std::unique_ptr<MockVLLMServer> server_;
    std::unique_ptr<AsyncLLMClient> client_;
};

// -- 1. Smoke ---------------------------------------------------------------

TEST_F(AsyncLLMClientTest, Smoke_OneRequestRoundtrip) {
    const std::string body = R"({"id":"chatcmpl-1","object":"chat.completion"})";
    server_->Start([&](const MockVLLMServer::Request&) {
        MockVLLMServer::Response r;
        r.status = 200;
        r.body = body;
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    std::promise<AsyncLLMClient::Response> p;
    auto f = p.get_future();
    client_->Submit(server_->Url("/v1/chat/completions"), R"({"hi":1})", /*gen=*/1, /*rid=*/"req-1",
                    [&p](AsyncLLMClient::Response r) { p.set_value(std::move(r)); });

    ASSERT_EQ(f.wait_for(5s), std::future_status::ready);
    auto r = f.get();
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.http_status, 200);
    EXPECT_EQ(r.body, body);
    EXPECT_TRUE(r.error.empty());
    EXPECT_GT(r.latency_us, 0);
}

// -- 2. Concurrent submission -----------------------------------------------

TEST_F(AsyncLLMClientTest, Concurrent_1000Callbacks_AllFireExactlyOnce) {
    server_->Start([](const MockVLLMServer::Request&) {
        MockVLLMServer::Response r;
        r.status = 200;
        r.body = R"({"ok":true})";
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    constexpr int kThreads = 100;
    constexpr int kPerThread = 10;
    constexpr int kTotal = kThreads * kPerThread;

    std::atomic<int> done_count{0};
    std::vector<std::atomic<int>> per_id_count(kTotal);
    for (auto& c : per_id_count) c.store(0);

    auto submitter = [&](int t) {
        for (int i = 0; i < kPerThread; ++i) {
            int idx = t * kPerThread + i;
            client_->Submit(server_->Url("/v1/chat/completions"),
                            R"({"x":1})",
                            /*gen=*/1,
                            /*rid=*/"r-" + std::to_string(idx),
                            [&, idx](AsyncLLMClient::Response r) {
                                per_id_count[idx].fetch_add(1, std::memory_order_relaxed);
                                if (r.ok) done_count.fetch_add(1, std::memory_order_relaxed);
                            });
        }
    };

    std::vector<std::thread> ts;
    ts.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) ts.emplace_back(submitter, t);
    for (auto& t : ts) t.join();

    ASSERT_TRUE(WaitFor(30s, [&] {
        return done_count.load(std::memory_order_relaxed) == kTotal;
    })) << "done_count=" << done_count.load();

    for (int i = 0; i < kTotal; ++i) {
        EXPECT_EQ(per_id_count[i].load(), 1) << "id " << i << " fired " << per_id_count[i].load() << " times";
    }
}

// -- 3. Cancellation of in-flight (mid-server-sleep) drops callback ---------

TEST_F(AsyncLLMClientTest, Cancel_InFlight_DropsCallback) {
    server_->Start([](const MockVLLMServer::Request&) {
        MockVLLMServer::Response r;
        r.status = 200;
        r.body = R"({"ok":true})";
        r.delay = 500ms;
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    std::atomic<bool> fired{false};
    client_->Submit(server_->Url("/v1/chat/completions"), R"({})", /*gen=*/42, "rid-cancel",
                    [&](AsyncLLMClient::Response) { fired.store(true); });

    std::this_thread::sleep_for(50ms);
    client_->CancelByGeneration(42);

    // Wait past the server's 500ms delay + some slack so the response would
    // have come back; assert no callback fired.
    std::this_thread::sleep_for(800ms);
    EXPECT_FALSE(fired.load());

    // Sanity: the server did process the request (so curl resources DID get
    // cleaned up, but the callback was suppressed). Not a leak check by
    // itself — leaks would be caught by ASan/leak sanitizer if enabled.
    EXPECT_GE(server_->Requests(), 1);
}

// -- 4. Cancellation of queued (not-yet-submitted) requests -----------------

TEST_F(AsyncLLMClientTest, Cancel_Queued_DroppedBeforeSubmit) {
    server_->Start([](const MockVLLMServer::Request&) {
        MockVLLMServer::Response r;
        r.status = 200;
        r.body = R"({"ok":true})";
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    constexpr int kFlood = 5000;
    constexpr uint64_t kGen = 99;
    std::atomic<int> fired{0};

    // Cancel BEFORE the wakeups can be drained, by interleaving the cancel
    // with the submit storm. The IO thread can only process inbox items at
    // the top of its loop iteration, so as long as the cancel insertion races
    // with the submitter, many items will be in inbox when the IO thread
    // first checks IsDead per item.
    std::thread canceller([&] {
        std::this_thread::sleep_for(1ms);
        client_->CancelByGeneration(kGen);
    });

    for (int i = 0; i < kFlood; ++i) {
        client_->Submit(server_->Url("/v1/chat/completions"),
                        R"({"x":1})",
                        /*gen=*/kGen,
                        /*rid=*/"q-" + std::to_string(i),
                        [&](AsyncLLMClient::Response) {
                            fired.fetch_add(1, std::memory_order_relaxed);
                        });
    }
    canceller.join();

    // Give the IO thread time to drain whatever made it past the cancel.
    std::this_thread::sleep_for(2s);

    // The cancellation mechanism must drop SOME requests before they hit the
    // mock server. We can't pin the exact number (it depends on scheduling),
    // but it must be strictly less than the full flood — otherwise cancel did
    // nothing for queued items.
    EXPECT_LT(server_->Requests(), kFlood)
        << "cancel never raced the inbox drain: server saw all "
        << kFlood << " requests";
    EXPECT_LT(fired.load(), kFlood) << "no callbacks were suppressed";
}

// -- 5. HTTP 500 error ------------------------------------------------------

TEST_F(AsyncLLMClientTest, HttpError_500_PopulatesResponse) {
    const std::string body = R"({"error":"boom"})";
    server_->Start([&](const MockVLLMServer::Request&) {
        MockVLLMServer::Response r;
        r.status = 500;
        r.body = body;
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    std::promise<AsyncLLMClient::Response> p;
    auto f = p.get_future();
    client_->Submit(server_->Url("/v1/chat/completions"), R"({})", 1, "rid-500",
                    [&](AsyncLLMClient::Response r) { p.set_value(std::move(r)); });

    ASSERT_EQ(f.wait_for(5s), std::future_status::ready);
    auto r = f.get();
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.http_status, 500);
    EXPECT_FALSE(r.error.empty());
    EXPECT_EQ(r.body, body);
}

// -- 6. Timeout (CURLOPT_TIMEOUT_MS = 60s fires) ----------------------------
//
// 192.0.2.0/24 is TEST-NET-1 (RFC 5737): routable-by-spec, but no host ever
// answers. libcurl waits the full CURLOPT_TIMEOUT_MS before giving up, so this
// exercises the timeout path end-to-end. The test is slow (~60s) but matches
// the spec requirement of validating the wall-clock timeout.
TEST_F(AsyncLLMClientTest, Timeout_PopulatesErrorAroundSixtySeconds) {
    client_ = std::make_unique<AsyncLLMClient>();

    std::promise<AsyncLLMClient::Response> p;
    auto f = p.get_future();
    client_->Submit("http://192.0.2.1:1/v1/chat/completions",
                    R"({})", 1, "rid-timeout",
                    [&](AsyncLLMClient::Response r) { p.set_value(std::move(r)); });

    ASSERT_EQ(f.wait_for(70s), std::future_status::ready);
    auto r = f.get();
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.http_status, 0);
    EXPECT_FALSE(r.error.empty());
    // Hard-coded CURLOPT_TIMEOUT_MS = 60000. Allow a generous lower bound; the
    // upper bound is the future's 70s wait above.
    EXPECT_GE(r.latency_us, 55LL * 1000 * 1000);
}

// -- 7. Connection reuse via CURLSH -----------------------------------------

TEST_F(AsyncLLMClientTest, ConnectionReuse_OneTcpConnectionForManySequential) {
    server_->Start([](const MockVLLMServer::Request&) {
        MockVLLMServer::Response r;
        r.status = 200;
        r.body = R"({"ok":1})";
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    constexpr int kRequests = 50;
    for (int i = 0; i < kRequests; ++i) {
        std::promise<void> p;
        auto f = p.get_future();
        client_->Submit(server_->Url("/v1/chat/completions"),
                        R"({"i":1})", 1, "rid-r-" + std::to_string(i),
                        [&](AsyncLLMClient::Response r) {
                            EXPECT_TRUE(r.ok);
                            p.set_value();
                        });
        ASSERT_EQ(f.wait_for(5s), std::future_status::ready);
    }

    EXPECT_EQ(server_->Requests(), kRequests);
    EXPECT_EQ(server_->Connections(), 1)
        << "expected single TCP connection reused; saw "
        << server_->Connections();
}

// -- 8. Destructor safety ---------------------------------------------------

TEST_F(AsyncLLMClientTest, Destructor_ClearsInFlightWithoutHangOrUAF) {
    server_->Start([](const MockVLLMServer::Request&) {
        MockVLLMServer::Response r;
        r.status = 200;
        r.body = R"({"ok":1})";
        r.delay = 200ms;  // ensure many are still in-flight at destruction
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    std::atomic<int> fired{0};
    constexpr int kReqs = 20;
    for (int i = 0; i < kReqs; ++i) {
        client_->Submit(server_->Url("/v1/chat/completions"), R"({})", 1,
                        "rid-d-" + std::to_string(i),
                        [&](AsyncLLMClient::Response) {
                            fired.fetch_add(1, std::memory_order_relaxed);
                        });
    }

    // Destroy immediately. Should: stop the IO thread, suppress any
    // post-destruction callbacks, and free curl handles cleanly.
    auto start = std::chrono::steady_clock::now();
    client_.reset();
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Destruction must not hang.
    EXPECT_LT(elapsed, 5s);

    // Some callbacks may have fired before the destructor took the IO thread
    // down; the contract is just "no UAF, no hang, no leak". Don't assert
    // fired==0 — that's an over-spec.
    SUCCEED();
}

// -- 9. X-Request-ID stamping -----------------------------------------------

TEST_F(AsyncLLMClientTest, XRequestId_StampedOnHeader) {
    std::string seen_rid;
    std::mutex m;
    server_->Start([&](const MockVLLMServer::Request& req) {
        std::lock_guard<std::mutex> lock(m);
        auto it = req.headers.find("x-request-id");
        if (it != req.headers.end()) seen_rid = it->second;
        MockVLLMServer::Response r;
        r.status = 200;
        r.body = R"({"ok":1})";
        return r;
    });
    client_ = std::make_unique<AsyncLLMClient>();

    std::promise<void> p;
    auto f = p.get_future();
    const std::string rid = "abc-123-XYZ";
    client_->Submit(server_->Url("/v1/chat/completions"), R"({})", 1, rid,
                    [&](AsyncLLMClient::Response r) {
                        EXPECT_TRUE(r.ok);
                        p.set_value();
                    });
    ASSERT_EQ(f.wait_for(5s), std::future_status::ready);

    std::lock_guard<std::mutex> lock(m);
    EXPECT_EQ(seen_rid, rid);
}

}  // namespace
}  // namespace flock

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
