#include <iostream>
#include <stack>

using namespace std;

class Editor {
public:
    string GetSelection() { return text; };
    void DeleteSelection() { text = ""; };
    void ReplaceSelection(const string& t) { text = t; };

private:
    string text;
};

class Application;
class Command {
public:
    Command(Application* app, Editor* editor) {
        this->app = app;
        this->editor = editor;
    }

    void SaveBackup() {
        backup = editor->GetSelection();
    }

    void Undo() {
        editor->ReplaceSelection(backup);
    }

    virtual bool Execute() = 0;

protected:
    Application* app;
    Editor* editor;
    string backup;
};

class CopyCommand : public Command {
public:
    bool Execute() override {
        SaveBackup();
        // app->clipboard = editor->GetSelection();
        return false;
    }
};

class CutCommand : public Command {
public:
    bool Execute() override {
        SaveBackup();
        // app->clipboard = editor->GetSelection();
        editor->DeleteSelection();
        return true;
    }
};

class PasteCommand : public Command {
public:
    bool Execute() override {
        SaveBackup();
        // editor->ReplaceSelection(app->clipboard);
        return true;
    }
};

class UndoCommand : public Command {
public:
    bool Execute() override {
        // app->Undo();
        return false;
    }
};

class CommandHistory {
public:
    void Push(Command* cmd) {
        history.push(cmd);
    }

    Command* Pop() {
        auto cmd = history.top();
        history.pop();
        return cmd;
    }

private:
    stack<Command*> history;
};

class Application {
    // omit cause one cpp file
};

int main(int argc, char *argv[]) {
    cout << "Command pattern needs to be complemented." << endl;
    cout << "One cpp file cannot satisfy." << endl;

    return 0;
}