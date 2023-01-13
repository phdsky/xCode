#include <iostream>
#include <vector>

using namespace std;

class Profile {
public:
    Profile (int index, string email) {
        id = index;
        content = email;
    }

    void GetId() {
        cout << "Id get! " << id << endl;
    }

    void GetEmail() {
        cout << "Email get! " << content << endl;
    }

private:
    int id;
    string content;
};

class ProfileIterator {
public:
    virtual ~ProfileIterator() = default;
    virtual Profile* GetNext() = 0;
    virtual bool HasMore() = 0;
};

class SocialNetwork {
public:
    virtual ProfileIterator* CreateFriendsIterator(int profileId) = 0;
    virtual ProfileIterator* CreateCoworkersIterator(int profileId) = 0;
};


class WeChat;
class WeChatIterator : public ProfileIterator {
public:
    WeChatIterator(WeChat* weChat, int profileId, const string& type) {
        this->weChat = weChat;
        this->profileId = profileId;
        this->type = type;
    };

    Profile* GetNext() override {
        if (HasMore()) {
            currentPosition++;
            return profiles[currentPosition - 1];
        }

        return nullptr;
    }

    bool HasMore() override {
        LazyInit();
        return currentPosition < profiles.size();
    }

private:
    WeChat* weChat;
    int profileId;
    string type;
    int currentPosition{0};
    vector<Profile*> profiles;

    void LazyInit() {
        if (profiles.empty()) {
            cout << "WeChatIterator lazy init." << endl;
            profiles.emplace_back(new Profile(profileId, type));
        }
    }
};

class WeChat : public SocialNetwork {
public:
    ProfileIterator* CreateFriendsIterator(int profileId) override {
        return new WeChatIterator(this, profileId, "friends");
    }

    ProfileIterator* CreateCoworkersIterator(int profileId) override {
        return new WeChatIterator(this, profileId, "coworkers");
    }
};

class SocialSpammer {
public:
    static void Send(ProfileIterator* iterator, const string& message) {
        while (iterator->HasMore()) {
            auto profile = iterator->GetNext();
            profile->GetId();
            profile->GetEmail();
        }
    }
};

int main(int argc, char *argv[]) {
    auto network = new WeChat();
    auto coworkerIter = network->CreateCoworkersIterator(0);
    auto friendIter = network->CreateFriendsIterator(1);

    SocialSpammer::Send(coworkerIter, "VIP MSG!");
    SocialSpammer::Send(friendIter, "JUNK MSG!");

    delete network;
    delete coworkerIter;
    delete friendIter;

    return 0;
}