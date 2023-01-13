#include <iostream>

using namespace std;

class Device {
public:
    virtual ~Device() = default;;
    virtual bool IsEnabled() const = 0;
    virtual void Enable() = 0;
    virtual void Disable() = 0;
    virtual int GetVolume() const = 0;
    virtual void SetVolume(int volume) = 0;
    virtual int GetChannel() const = 0;
    virtual void SetChannel(int channel) = 0;
};

class Tv : public Device {
public:
    bool IsEnabled() const override {
        return power;
    }

    void Enable() override {
        cout << "Turn ON the tv.\n";
        power = true;
    }

    void Disable() override {
        cout << "Turn OFF the tv.\n";
        power = false;
    }

    int GetVolume() const override {
        return volume;
    }

    void SetVolume(int v) override {
        volume = v;
        cout << "Set Tv volume to " << volume << endl;
    }

    int GetChannel() const override {
        return channel;
    }

    void SetChannel(int c) override {
        channel = c;
        cout << "Set Tv channel to " << channel << endl;
    }

private:
    bool power;
    int volume;
    int channel;
};

class Radio : public Device {
public:
    bool IsEnabled() const override {
        return power;
    }

    void Enable() override {
        cout << "Turn ON the Radio.\n";
        power = true;
    }

    void Disable() override {
        cout << "Turn OFF the Radio.\n";
        power = false;
    }

    int GetVolume() const override {
        return volume;
    }

    void SetVolume(int v) override {
        volume = v;
        cout << "Set Radio volume to " << volume << endl;
    }

    int GetChannel() const override {
        return channel;
    }

    void SetChannel(int c) override {
        channel = c;
        cout << "Set Radio channel to " << channel << endl;
    }
private:
    bool power;
    int volume;
    int channel;
};

class RemoteControl {
public:
    explicit RemoteControl(Device* device) {
        this->device = device;
    }

    virtual void TogglePower() const {
        if (device->IsEnabled()) {
            device->Disable();
        } else {
            device->Enable();
        }
    }

    virtual void VolumeDown() const {
        device->SetVolume(device->GetVolume() - 10);
    }

    virtual void VolumeUp() const {
        device->SetVolume(device->GetVolume() + 10);
    }

    virtual void ChannelDown() const {
        device->SetChannel(device->GetChannel() - 1);
    }

    virtual void ChannelUp() const {
        device->SetChannel(device->GetChannel() + 1);
    }

protected:
    Device *device;
};

class AdvancedRemoteControl : public RemoteControl {
public:
    explicit AdvancedRemoteControl(Device *device) : RemoteControl(device) {}

    void Mute() {
        device->SetVolume(0);
    }
};

int main(int argc, char* argv[]) {
    auto tv = new Tv();
    auto remote = new RemoteControl(tv);
    remote->TogglePower();

    auto radio = new Radio();
    auto advRemote = new AdvancedRemoteControl(radio);
    advRemote->Mute();

    delete advRemote;
    delete radio;

    delete remote;
    delete tv;

    return 0;
}