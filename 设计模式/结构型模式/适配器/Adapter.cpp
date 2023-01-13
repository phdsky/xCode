#include <iostream>
#include <math.h>

using namespace std;

class RoundPeg {
public:
    RoundPeg() = default;
    explicit RoundPeg(int r) : radius(r) {}

    virtual ~RoundPeg() = default;

    virtual int GetRadius() const {
        return radius;
    }

private:
    int radius;
};

class RoundHole {
public:
    explicit RoundHole(int r) : radius(r) {}
    virtual ~RoundHole() = default;

    virtual int GetRadius() const {
        return radius;
    }

    string Fits(const RoundPeg* roundPeg) const {
        if (this->GetRadius() >= roundPeg->GetRadius()) {
            return "True\n";
        } else {
            return "False\n";
        }
    }

private:
    int radius;
};

class SquarePeg {
public:
    explicit SquarePeg(int w) : width(w) {}
    ~SquarePeg() = default;

    virtual int GetWidth() const {
        return width;
    }

private:
    int width;
};

class SquarePegAdapter : public RoundPeg {
public:
    explicit SquarePegAdapter(SquarePeg* squarePeg) {
        this->squarePeg = squarePeg;
    }

    int GetRadius() const override {
        return squarePeg->GetWidth() * sqrt(2) / 2;
    }

private:
    SquarePeg *squarePeg;
};

int main(int argc, char *argv[]) {
    auto roundHole = new RoundHole(5);

    auto roundPeg = new RoundPeg(5);
    cout << roundHole->Fits(roundPeg);
    delete roundPeg;

    auto sSquarePeg = new SquarePeg(5);
    auto lSquarePeg = new SquarePeg(10);
    // cout << roundHole->Fits(sSquarePeg); // type is diff

    auto sSquarePegAdapter = new SquarePegAdapter(sSquarePeg);
    auto lSquarePegAdapter = new SquarePegAdapter(lSquarePeg);
    cout << roundHole->Fits(sSquarePegAdapter);
    cout << roundHole->Fits(lSquarePegAdapter);

    delete sSquarePegAdapter;
    delete lSquarePegAdapter;

    delete sSquarePeg;
    delete lSquarePeg;

    delete roundHole;

    return 0;
}