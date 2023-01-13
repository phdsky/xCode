#include <iostream>
#include <vector>

using namespace std;

class Graphic {
public:
    virtual ~Graphic() = default;
    virtual void Move(int x, int y) = 0;
    virtual void Draw() const = 0;
};

class Dot : public Graphic {
public:
    Dot(int x, int y) {
        posX = x;
        posY = y;
    }

    void Move(int x, int y) override {
        cout << "Move dot from (" << posX << ", " << posY << ") ";
        posX += x; posY += y;
        cout << "to (" << posX << ", " << posY << ")" << endl;
    }

    void Draw() const override {
        cout << "Draw dot at (" << posX << ", " << posY << ")." << endl;
    }

protected:
    int posX;
    int posY;
};

class Circle : public Dot {
public:
    Circle(int x, int y, int r) : Dot(x, y) {
        radius = r;
    }

    void Draw() const override {
        cout << "Draw circle at (" << posY << ", " << posY << ")." <<
        " radius: " << radius << endl;
    }

private:
    int radius;
};

class CompoundGraphic : public Graphic {
public:
    void Add(Graphic* graphic) {
        children.emplace_back(graphic);
    }

    void Remove(Graphic* graphic) {
        if (!children.empty()) {
            children.pop_back();
        }
    }

    void Move(int x, int y) override {
        for (auto& child : children) {
            child->Move(x, y);
        }
    }

    void Draw() const override {
        for (auto& child : children) {
            child->Draw();
        }
    }

private:
    vector<Graphic *> children;
};

class ImageEditor {
public:
    ImageEditor () {
        Load();
    }

    ~ImageEditor() {
        delete compoundGraphic;
    }

    void Load() {
        compoundGraphic = new CompoundGraphic();
        compoundGraphic->Add(new Dot(1, 2));
        compoundGraphic->Add(new Circle(5, 3, 10));
    }

    void GroupSelected() {
        auto group = new CompoundGraphic();
        group->Add(new Dot(2, 3));
        group->Add(new Dot(3, 2));
        group->Add(new Circle(2, 3, 1));

        compoundGraphic->Add(group);
        compoundGraphic->Draw();

        delete group;
    }

private:
    CompoundGraphic* compoundGraphic{};
};

int main(int argc, char *argv[]) {
    auto imageEditor = new ImageEditor();
    imageEditor->GroupSelected();

    delete imageEditor;

    return 0;
}