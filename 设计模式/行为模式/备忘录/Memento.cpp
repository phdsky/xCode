#include <iostream>

using namespace std;

class Snapshot;
class Editor {
public:
    void SetText(const string& t) {
        text = t;
    }

    void SetCursor(int x, int y) {
        this->curX = x;
        this->curY = y;
    }

    void SetSelectionWidth(int width) {
        selectionWidth = width;
    }

    Snapshot* CreateSnapshot(const string& t, int x, int y, int w) {
        cout << "Create Snapshot." << endl;
        return nullptr;
        // return new Snapshot(this, t, x, y, w); // cannot implement in one cpp file
    }

private:
    string text;
    int curX{}, curY{};
    int selectionWidth{};
};

class Snapshot {
public:
    Snapshot(Editor* e, const string& t, int x, int y, int w) {
        editor = e;
        text = t;
        curX = x;
        curY = y;
        selectionWidth = w;
    };

    void Restore() {
        cout << "Restore from snapshot." << endl;
        editor->SetText(text);
        editor->SetCursor(curX, curY);
        editor->SetSelectionWidth(selectionWidth);
    }

private:
    Editor* editor;
    string text;
    int curX, curY;
    int selectionWidth;
};

class Command {
public:
    void MakeBackup(Editor* editor, const string& t, int x, int y, int w) {
        backup = editor->CreateSnapshot(t, x, y, w);
    }

    void Undo() {
        if (backup != nullptr) {
            backup->Restore();
        }
    }

private:
    Snapshot* backup;
};


int main(int argc, char *argv[]) {
    auto editor = new Editor();
    auto command = new Command();

    command->MakeBackup(editor, "Older State", 1, 2, 3);
    editor->SetText("Newer State");
    editor->SetCursor(2, 3);
    command->Undo();

    delete command;
    delete editor;

    return 0;
}