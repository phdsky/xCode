#include <iostream>
#include <unordered_map>
#include <utility>

using namespace std;

class Shape {
public:
    virtual ~Shape() = default;

    Shape(string proto) : name(std::move(proto)) {};

    virtual Shape *Clone() const = 0;

    virtual void Method(int xx, int yy, string cc) {
        this->x = xx;
        this->y = yy;
        this->color = move(cc);
        cout << "Call Method from " << name << " with field:\n"
             << "x: " << x << " y: " << y << " color: " << color << endl;
    }

protected:
    int x;
    int y;
    string color;
    string name;
};

class Rectangle : public Shape {
public:
    Rectangle(string name, int w, int h) :
    Shape(name), width(w), height(h) {};

    Shape* Clone() const override {
        return new Rectangle(*this);
    }

private:
    int width;
    int height;
};

class Circle : public Shape {
public:
    Circle(string name, int r) :
    Shape(name), radius(r) {};

    Shape* Clone() const override {
        return new Circle(*this);
    }

private:
    int radius;
};

class Application {
public:
    Application() {
        prototypes["rectangle"] = new Rectangle("rectangle", 10, 20);;
        prototypes["circle"] = new Circle("circle", 30);
    }

    ~Application() {
        delete prototypes["rectangle"];
        delete prototypes["circle"];
    }

    Shape *CreatePrototype(string type) {
        return prototypes[type]->Clone();
    }

private:
    unordered_map<string, Shape*> prototypes;
};

int main(int argc, char *argv[]) {
    Application app = Application();

    Shape *rectangle = app.CreatePrototype("rectangle");
    rectangle->Method(11, 22, "pink");
    delete rectangle;

    Shape *circle = app.CreatePrototype("circle");
    circle->Method(33, 44, "yellow");
    delete circle;

    return 0;
}