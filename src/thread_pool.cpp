#include "thread_pool.hpp"
#include <utility>

ThreadPool::ThreadPool(std::size_t num_threads)
    : stop(false), tasks_in_progress(0)
{
    if (num_threads == 0)
    {
        num_threads = 1;
    }

    for (std::size_t i = 0; i < num_threads; i++)
    {
        workers.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }

    task_cv.notify_all();

    for (std::size_t i = 0; i < workers.size(); i++)
    {
        if (workers[i].joinable())
        {
            workers[i].join();
        }
    }
}

void ThreadPool::enqueue(std::function<void()> task)
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.push(std::move(task));
    }

    task_cv.notify_one();
}

void ThreadPool::wait_for_all()
{
    std::unique_lock<std::mutex> lock(queue_mutex);

    finished_cv.wait(lock, [this]()
    {
        return tasks.empty() && tasks_in_progress == 0;
    });
}

void ThreadPool::worker_loop()
{
    while (true)
    {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            task_cv.wait(lock, [this]()
            {
                return stop || !tasks.empty();
            });

            if (stop && tasks.empty())
            {
                return;
            }

            task = std::move(tasks.front());
            tasks.pop();
            tasks_in_progress++;
        }

        task();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks_in_progress--;

            if (tasks.empty() && tasks_in_progress == 0)
            {
                finished_cv.notify_all();
            }
        }
    }
}