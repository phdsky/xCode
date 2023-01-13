#include <iostream>
#include <unordered_map>

using namespace std;

class EventListener {
public:
    virtual void Update(const string& filename) = 0;
};

class LoggingListener : public EventListener {
public:
    LoggingListener(const string& filename, const string& message) {
        log = filename;
        msg = message;
    }

    void Update(const string& filename) override {
        cout << "Logging: " << log << ", Content:" << msg << endl;
        log = filename;
    }

private:
    string log;
    string msg;
};

class EmailAlertsListener : public EventListener {
public:
    EmailAlertsListener(const string& email, const string& message) {
        eml = email;
        msg = message;
    }

    void Update(const string& filename) override {
        cout << "EmailAlert: " << eml << ", Content:" << msg << endl;
        eml = filename;
    }

private:
    string eml;
    string msg;
};

class EventManger {
public:
    void Subscribe(const string& eventType, EventListener* listener) {
        listeners[eventType] = listener;
    }

    void Unsubscribe(const string& eventType, EventListener* listener) {
        listeners.erase(eventType);
    }

    void Notify(const string& eventType, const string& data) {
        for (const auto& listener : listeners) {
            listener.second->Update(data);
        }
    }

private:
    unordered_map<string, EventListener*> listeners;
};

class Editor {
public:
    Editor() {
        events = new EventManger();
    }

    void OpenFile(const string& path) {
        file = path;
        events->Notify("open", file);
    }

    void SaveFile() {
        events->Notify("save", file);
    }

    EventManger* events;

private:
    string file;
};

int main(int argc, char *argv[]) {
    auto editor = new Editor();

    auto logger = new LoggingListener("/path/to/log.txt", "有人打开了文件");
    editor->events->Subscribe("open", logger);

//    auto emailAlerts = new EmailAlertsListener("admin@example.com", "有人更改了文件");
//    editor->events->Subscribe("save", emailAlerts);

    editor->OpenFile("JayChou.album");
    editor->SaveFile();

//    delete emailAlerts;
    delete logger;
    delete editor;

    return 0;
}