#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

class TreeType {
public:
    TreeType(const string& n, const string& c, const string& t) {
        name = n;
        color = c;
        texture = t;
    }

    void Draw(int x, int y) {
        cout << "Draw " << "(" << name << ", " << color << ", " << texture
        << ")" << " Tree at (" << x << ", " << y << ")." << endl;
    }

private:
    string name;
    string color;
    string texture;
};

class TreeFactory {
public:
    TreeType* GetTreeType(const string& name, const string& color, const string& texture) {
        auto type = treeTypes.find(name + color + texture);
        TreeType* treeType = nullptr;
        if (type == treeTypes.end()) {
            treeType = new TreeType(name, color, texture);
            treeTypes[name + color + texture] = treeType;
        } else {
            treeType = type->second;
        }

        return treeType;
    }

private:
    unordered_map<string, TreeType*> treeTypes;
};

class Tree {
public:
    Tree(int x, int y, TreeType* t) {
        this->posX = x;
        this->posY = y;
        this->type = t;
    }

    void Draw() {
        type->Draw(posX, posY);
    }

private:
    int posX;
    int posY;
    TreeType* type;
};

class Forest {
public:
    void PlantTree(int x, int y, const string& name, const string& color, const string& texture) {
        TreeType* type = (new TreeFactory)->GetTreeType(name, color, texture);
        Tree* tree = new Tree(x, y, type);
        trees.emplace_back(tree);
    }

    void Draw() {
        for (const auto& tree : trees) {
            tree->Draw();
        }
    }

private:
    vector<Tree*> trees;
};

int main(int argc, char *argv[]) {
    auto forest = new Forest();
    forest->PlantTree(1, 1, "ZhangShu", "Green", "Shrink");
    forest->PlantTree(3, 4, "Feng", "Yellow", "Bling");
    forest->Draw();

    delete forest;

    return 0;
}