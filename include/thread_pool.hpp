#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <cstddef>

class ThreadPool
{
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable task_cv;
    std::condition_variable finished_cv;
    bool stop;
    int tasks_in_progress;

    void worker_loop();

public:
    ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    void enqueue(std::function<void()> task);
    void wait_for_all();
};