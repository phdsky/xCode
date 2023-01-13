#include <iostream>

using namespace std;

class AudioPlayer;
class State {
public:
    ~State() = default;
    explicit State(AudioPlayer* audioPlayer) {
        player = audioPlayer;
    }

    virtual void ClickLock() = 0;
    virtual void ClickPlay() = 0;
    virtual void ClickNext() = 0;
    virtual void ClickPrevious() = 0;

protected:
    AudioPlayer* player;
};

class LockedState : public State {
public:
    void ClickLock() override {}
    void ClickPlay() override {}
    void ClickNext() override {}
    void ClickPrevious() override {}
};

class ReadyState : public State {};

class PlayingState : public State {};

class AudioPlayer {
public:

private:
    State* state;
};

int main(int argc, char *argv[]) {
    cout << "State pattern needs to be complemented." << endl;
    cout << "One cpp file cannot satisfy." << endl;

    return 0;
}