#include <iostream>

using namespace std;

class Dot;
class Circle;
class Rectangle;
class CompoundShape;

class Visitor {
public:
    ~Visitor() = default;
    virtual void VisitDot(Dot* dot) = 0;
    virtual void VisitCircle(Circle* circle) = 0;
    virtual void VisitRectangle(Rectangle* rectangle) = 0;
    virtual void VisitCompoundShape(CompoundShape* compoundShape) = 0;
};

class Shape {
public:
    ~Shape() = default;
    // virtual void Move(int x, int y) = 0;
    // virtual void Draw() const = 0;
    virtual void Accept(Visitor* v) = 0;
};

class Dot : public Shape {
public:

    void Accept(Visitor* v) override {
        v->VisitDot(this);
    }
};

class Circle : public Shape {
public:
    void Accept(Visitor* v) override {
        v->VisitCircle(this);
    }
};

class Rectangle : public Shape {
public:
    void Accept(Visitor* v) override {
        v->VisitRectangle(this);
    }
};

class CompoundShape : public Shape {
public:
    void Accept(Visitor* v) override {
        v->VisitCompoundShape(this);
    }
};

class XMLExportVisitor : public Visitor {
public:
    void VisitDot(Dot* dot) override {
        cout << "Export dot XML." << endl;
    }

    void VisitCircle(Circle* circle) override {
        cout << "Export circle XML." << endl;
    }

    void VisitRectangle(Rectangle* rectangle) override {
        cout << "Export rectangle XML." << endl;
    }

    void VisitCompoundShape(CompoundShape* compoundShape) override {
        cout << "Export compound XML." << endl;
    }
};

int main(int argc, char *argv[]) {
    auto circle = new Circle();
    auto exportVisitor = new XMLExportVisitor();

    circle->Accept(exportVisitor);

    delete exportVisitor;
    delete circle;

    return 0;
}