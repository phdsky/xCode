#include <iostream>

using namespace std;

class DataSource {
public:
    virtual ~DataSource() = default;
    virtual void WriteData() const = 0;
    virtual void ReadData() const = 0;
};

class FileDataSource : public DataSource {
public:
    FileDataSource() = default;

    void WriteData() const override {
        cout << "Write File Data" << endl;
    }

    void ReadData() const override {
        cout << "Read File Data" << endl;
    }
};

class DataSourceDecorator : public DataSource {
public:
    explicit DataSourceDecorator(DataSource* dataSource) {
        wrappee = dataSource;
    }

    void WriteData() const override {
        wrappee->WriteData();
    }

    void ReadData() const override {
        wrappee->ReadData();
    }

protected:
    DataSource* wrappee;
};

class EncryptionDecorator : public DataSourceDecorator {
public:
    explicit EncryptionDecorator(DataSource* dataSource) :
    DataSourceDecorator(dataSource) {}

    void WriteData() const override {
        cout << "Encryption Write" << endl;
        DataSourceDecorator::WriteData();
    }

    void ReadData() const override {
        cout << "Encryption Read" << endl;
        DataSourceDecorator::ReadData();
    }
};

class CompressionDecorator : public DataSourceDecorator {
public:
    explicit CompressionDecorator(DataSource* dataSource) :
    DataSourceDecorator(dataSource) {};

    void WriteData() const override {
        cout << "Compression Write" << endl;
        DataSourceDecorator::WriteData();
    }

    void ReadData() const override {
        cout << "Compression Read" << endl;
        DataSourceDecorator::ReadData();
    }
};

int main(int argc, char *argv[]) {
    auto fileDataSource = new FileDataSource();
    fileDataSource->WriteData();

    auto compression = new CompressionDecorator(fileDataSource);
    compression->WriteData();

    auto encryption = new EncryptionDecorator(fileDataSource);
    encryption->WriteData();

    delete encryption;
    delete compression;
    delete fileDataSource;

    return 0;
}