#include <iostream>

using namespace std;

class Button {
public:
    virtual ~Button()= default;;
    virtual string Paint() const = 0;
};

class WinButton : public Button {
public:
    string Paint() const override {
        return "WinButton Paint\n";
    }
};

class MacButton : public Button {
public:
    string Paint() const override {
        return "MacButton Paint\n";
    }
};

class Checkbox {
public:
    virtual ~Checkbox()= default;;
    virtual string Paint() const = 0;
};

class WinCheckbox : public Checkbox {
public:
    string Paint() const override {
        return "WinCheckbox Paint\n";
    }
};

class MacCheckbox : public Checkbox {
public:
    string Paint() const override {
        return "MacCheckbox Paint\n";
    }
};

class GUIFactory {
public:
    virtual ~GUIFactory()= default;;
    virtual Button *CreateButton() const = 0;
    virtual Checkbox *CreateCheckbox() const = 0;
};

class WinFactory : public GUIFactory {
public:
    Button *CreateButton() const override {
        return new WinButton();
    }

    Checkbox *CreateCheckbox() const override {
        return new WinCheckbox();
    }
};

class MacFactory : public GUIFactory {
public:
    Button *CreateButton() const override {
        return new MacButton();
    }

    Checkbox *CreateCheckbox() const override {
        return new MacCheckbox();
    }
};

class Application {
public:
    explicit Application(GUIFactory* factory) {
        this->factory = factory;
    }

    void CreateUI() {
        button = factory->CreateButton();
        checkbox = factory->CreateCheckbox();
    }

    void Paint() {
        cout << button->Paint();
        cout << checkbox->Paint();
    }

private:
    GUIFactory* factory{};
    Button* button{};
    Checkbox *checkbox{};
};

int main(int argc, char *argv[]) {
    string config = "Windows";
    GUIFactory *factory;

    if (config == "Windows") {
        factory = new WinFactory();
    } else if (config == "Mac") {
        factory = new MacFactory();
    } else {
        cout << "Error! Unknown Operating System Type!" << endl;
        return -1;
    }

    Application app = Application(factory);
    app.CreateUI();
    app.Paint();

    delete factory;

    return 0;
}