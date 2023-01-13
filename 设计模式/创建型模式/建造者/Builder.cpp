#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Car {
public:
    vector<string> parts;

    void ListParts() const {
        cout << "Car parts: ";
        for (size_t i = 0; i < parts.size(); i++) {
            if (parts[i] == parts.back()) {
                cout << parts[i];
            } else {
                cout << parts[i] << ", ";
            }
        }
        cout << "\n\n";
    }
};

class Manual {
public:
    vector<string> parts;

    void ListParts() const {
        cout << "Manual parts: ";
        for (size_t i = 0; i < parts.size(); i++) {
            if (parts[i] == parts.back()) {
                cout << parts[i];
            } else {
                cout << parts[i] << ", ";
            }
        }
        cout << "\n\n";
    }
};

class Builder {
public:
    virtual ~Builder() = default;
    virtual void Reset() = 0;
    virtual void SetSeats(int num) const = 0;
    virtual void SetEngine(string type) const = 0;
    virtual void SetTripComputer() const = 0;
    virtual void SetGPS() const = 0;
};

class CarBuilder : public Builder {
public:
    CarBuilder() {
        Reset();
    }

    void Reset() override {
        car = new Car();
    }

    void SetSeats(int num) const override {
        car->parts.emplace_back("Seats " + to_string(num));
    }

    void SetEngine(string type) const override {
        car->parts.emplace_back("Engine " + type);
    }

    void SetTripComputer() const override {
        car->parts.emplace_back("TripComputer");
    }

    void SetGPS() const override {
        car->parts.emplace_back("GPS");
    }

    Car* GetProduct() {
        Car* product = car;
        Reset();
        return product;
    }

private:
    Car *car{};
};

class CarManualBuilder : public Builder {
public:
    CarManualBuilder() {
        Reset();
    }

    void Reset() override {
        manual = new Manual();
    }

    void SetSeats(int num) const override {
        manual->parts.emplace_back("Seats M " + to_string(num));
    }

    void SetEngine(string type) const override {
        manual->parts.emplace_back("Engine M " + type);
    }

    void SetTripComputer() const override {
        manual->parts.emplace_back("TripComputer M");
    }

    void SetGPS() const override {
        manual->parts.emplace_back("GPS M");
    }

    Manual* GetProduct() {
        Manual* product = manual;
        Reset();
        return product;
    }

private:
    Manual* manual{};
};

class Director {
public:
    void SetBuilder(Builder* builder) {
        this->builder = builder;
    }

    void ConstructSpotsCar() {
        builder->Reset();
        builder->SetSeats(2);
        builder->SetEngine("Sport");
        builder->SetGPS();
    }

    void ConstructSUV() {
        builder->Reset();
        builder->SetSeats(5);
        builder->SetEngine("SUV");
        builder->SetTripComputer();
        builder->SetGPS();
    }

private:
    Builder* builder;
};

int main(int argc, char *argv[]) {
    Director* director = new Director();

    // Build Cars
    CarBuilder* builder = new CarBuilder();
    director->SetBuilder(builder);

    cout << "Build Sports Car" << endl;
    director->ConstructSpotsCar();
    Car* car = builder->GetProduct();
    car->ListParts();
    delete car;

    cout << "Build SUV Car" << endl;
    director->ConstructSUV();
    car = builder->GetProduct();
    car->ListParts();

    delete builder;


    // Build Car Manuals
    CarManualBuilder* builderM = new CarManualBuilder();
    director->SetBuilder(builderM);

    cout << "Build Sports Car Manual" << endl;
    director->ConstructSpotsCar();
    Manual* manual = builderM->GetProduct();
    manual->ListParts();
    delete manual;

    cout << "Build SUV Car Manual" << endl;
    director->ConstructSUV();
    manual = builderM->GetProduct();
    manual->ListParts();

    delete builderM;
    delete director;

    return 0;
}