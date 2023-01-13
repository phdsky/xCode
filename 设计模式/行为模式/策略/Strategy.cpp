#include <iostream>

using namespace std;

class Strategy {
public:
    ~Strategy() = default;
    virtual int Execute(int a, int b) = 0;
};

class ConcreteStrategyAdd : public Strategy {
    int Execute(int a, int b) override {
        return a + b;
    }
};

class ConcreteStrategySubtract : public Strategy {
    int Execute(int a, int b) override {
        return a - b;
    }
};

class ConcreteStrategyMultiply : public Strategy {
    int Execute(int a, int b) override {
        return a * b;
    }
};

class Context {
public:
    void SetStrategy(Strategy* s) {
        strategy = s;
    }

    int ExecuteStrategy(int a, int b) {
        return strategy->Execute(a, b);
    }

private:
    Strategy* strategy;
};

int main(int argc, char *argv[]) {
    auto context = new Context();
    int a = 10, b = 25;

    string action = "add";
    if (action == "add") {
        context->SetStrategy(new ConcreteStrategyAdd());
    } else if (action == "subtraction") {
        context->SetStrategy(new ConcreteStrategySubtract());
    } else if (action == "multiplication") {
        context->SetStrategy(new ConcreteStrategyMultiply());
    } else {
        cout << "Wops! Error Action." << endl;
        return -1;
    }

    cout << context->ExecuteStrategy(a, b) << endl;

    delete context;

    return 0;
}