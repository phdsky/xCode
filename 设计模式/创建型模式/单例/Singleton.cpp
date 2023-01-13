#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

using namespace std;

class Singleton {
public:
    Singleton(Singleton &instance) = delete;
    void operator=(const Singleton&) = delete;

    static Singleton *GetInstance(const string& value);

    string GetValue() const {
        return value;
    }

protected:
    explicit Singleton(string val) : value(std::move(val)) {}
    ~Singleton() = default;
    string value;

private:
    static Singleton* instance;
    static std::mutex mutex;
};

// Static methods should be defined outside the class
Singleton* Singleton::instance{nullptr};
std::mutex Singleton::mutex;

Singleton *Singleton::GetInstance(const string &value) {
    std::lock_guard<std::mutex> lock(mutex);
    if (instance == nullptr) {
        instance = new Singleton(value);
    }

    return instance;
}

void ThreadFoo() {
    // Following code emulates slow initialization.
    std::this_thread::sleep_for(std::chrono::milliseconds(1001));
    Singleton* singleton = Singleton::GetInstance("FOO");
    cout << singleton->GetValue() << endl;
}

void ThreadBar() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    Singleton* singleton = Singleton::GetInstance("BAR");
    cout << singleton->GetValue() << endl;
}

int main(int argc, char *argv[]) {
    std::cout <<"If you see the same value, then singleton was reused (yay!\n" <<
              "If you see different values, then 2 singletons were created (booo!!)\n\n" <<
              "RESULT:\n";
    std::thread t1(ThreadFoo);
    std::thread t2(ThreadBar);
    t1.join();
    t2.join();

    return 0;
}