#include <iostream>

using namespace std;

class GameAI {
public:
    void Turn() {};
    void CollectResources() {};
    void Attack() {};

    virtual void BuildStructures() = 0;
    virtual void BuildUnits() = 0;

    virtual void SendScouts(int p) = 0;
    virtual void SendWarriors(int p) = 0;
};

class OrcsAI : public GameAI {
public:
    void BuildStructures() override {}
    void BuildUnits() override {}
    void SendScouts(int p) override {}
    void SendWarriors(int p) override {}
};

class MonstersAI : public GameAI {
    void CollectResources() {}
    void BuildStructures() override {}
    void BuildUnits() override {}
};

int main(int argc, char *argv[]) {
    cout << "Nothing to implement." << endl;

    return 0;
}