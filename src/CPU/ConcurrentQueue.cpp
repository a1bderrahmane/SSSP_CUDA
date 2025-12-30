/* Based on code from : https://cppscripts.com/cpp-concurrent-queue */

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ConcurrentQueue {
public:

    /** Atomically adds given value to the queue. */
    void enqueue(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        cond_var_.notify_one();
    }

    /** Atomically removes and returns the first element of the queue.
     * If the queue is empty, returns NULL.
     */
    T dequeue() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty) {
            return NULL;
        }
        T value = queue_.front();
        queue_.pop();
        return value;
    }

    /** Warning : not thread-safe, should be used by the main thread only */
    bool isEmpty() {
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};