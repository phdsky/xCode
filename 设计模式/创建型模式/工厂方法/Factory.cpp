#include <iostream>

using namespace std;

class Button {
public:
    virtual ~Button()= default;
    virtual string Render() const = 0;
    virtual string OnClick() const = 0;
};

class WindowsButton : public Button {
public:
    string Render() const override {
        return "Windows Render\n";
    }

    string OnClick() const override {
        return "Windows OnClick\n";
    }
};

class HTMLButton : public Button {
public:
    string Render() const override {
        return "HTML Render\n";
    }

    string OnClick() const override {
        return "HTML OnClick\n";
    }
};

class Dialog {
public:
    virtual ~Dialog() = default;
    virtual Button* CreateButton() const = 0;

    string Render() const {
        Button* button = this->CreateButton();
        return (button->OnClick() + button->Render());
    }
};

class WindowsDialog : public Dialog {
public:
    Button* CreateButton() const override {
        return new WindowsButton();
    }
};

class WebDialog : public Dialog {
public:
    Button* CreateButton() const override {
        return new HTMLButton();
    }
};

int main(int argc, char* argv[]) {
    string config = "Web";
    Dialog* dialog;

    if (config == "Windows") {
        dialog = new WindowsDialog();
    } else if (config == "Web") {
        dialog = new WebDialog();
    } else {
        cout << "Error! Unknown Operating System Type!" << endl;
        return -1;
    }

    cout << dialog->Render();

    delete dialog;

    return 0;
}