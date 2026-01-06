/* Based on code from : https://cppscripts.com/cpp-concurrent-queue */

#ifndef CONCURRENT_QUEUE_HPP
#define CONCURRENT_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include "CSR.hpp"

#define UINT_INFINITY 4294967295

/** A concurrent queue that can contain uint values. */
class ConcurrentQueue {
public:

    /** Atomically adds given value to the queue. */
    void enqueue(uint value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        cond_var_.notify_one();
    }

    /** Atomically removes and returns the first element of the queue.
     * If the queue is empty, returns UINT_INFINITY (2^32 - 1).
     */
    uint dequeue() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return UINT_INFINITY;
        }
        uint value = queue_.front();
        queue_.pop();
        return value;
    }

    /** Warning : not thread-safe, should be used by the main thread only */
    bool isEmpty() {
        return queue_.empty();
    }

private:
    std::queue<uint> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};

#endif // CONCURRENT_QUEUE_HPP