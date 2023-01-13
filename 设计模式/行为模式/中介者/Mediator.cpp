#include <iostream>

using namespace std;

class Component;
class Mediator {
public:
    virtual void Notify(Component* sender, const string& event) = 0;
};

class Component {
public:
    explicit Component(Mediator* mediator) {
        dialog = mediator;
    }

    void Click() {
        dialog->Notify(this, "click");
    }

    void KeyPress() {
        dialog->Notify(this, "keypress");
    }

protected:
    Mediator* dialog;
};

class Button : public Component {

};

class Textbox : public Component {

};

class Checkbox : public Component {
public:
    explicit Checkbox(Mediator* mediator) : Component(mediator) {};

    void Check() {
        dialog->Notify(this, "check");
    }
};

class AuthenticationDialog : public Mediator {
public:
    explicit AuthenticationDialog(const string& dialog) {
        title = dialog;
    }

    void Notify(Component* sender, const string& event) override {
        cout << "Auth " << event << endl;
    }

private:
    string title;
    // Checkbox* loginOrRegisterCheckbox;
    // Textbox* loginUserName, loginPassword;
    // Textbox* registrationUsername, registrationPassword, registrationEmail;
    // Button* okButton, cancelButton;
};

int main(int argc, char *argv[]) {
    auto auth = new AuthenticationDialog("OnBoarding");
    auto checkbox = new Checkbox(auth);

    auth->Notify(checkbox, "notify");
    checkbox->Check();

    delete checkbox;
    delete auth;

    return 0;
}