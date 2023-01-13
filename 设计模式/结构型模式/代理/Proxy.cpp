#include <iostream>

using namespace std;

class ThirdPartyTvLib {
public:
    virtual string ListVideos() = 0;
    virtual string GetVideoInfo(int id) = 0;
    virtual void DownloadVideo(int id) const = 0;
};

class ThirdPartyTvClass : public ThirdPartyTvLib {
public:
    string ListVideos() override {
        cout << "ThirdPartyTvClass list videos" << endl;
        return "xx video listed\n";
    }

    string GetVideoInfo(int id) override {
        cout << "ThirdPartyTvClass get info" << endl;
        return to_string(id) + " video info\n";
    }

    void DownloadVideo(int id) const override {
        cout << "Download " << id << " video" << endl;
    }
};

class CachedTvClass : public ThirdPartyTvLib {
public:
    CachedTvClass(ThirdPartyTvLib* thirdPartyTvLib) {
        service = thirdPartyTvLib;
    }

    string ListVideos() override {
        cout << "CachedTvClass list videos" << endl;
        if (listCache.empty() || needReset) {
            listCache = service->ListVideos();
        }

        return listCache;
    }

    string GetVideoInfo(int id) override {
        cout << "CachedTvClass get info" << endl;
        if (videoCache.empty() || needReset) {
            videoCache = service->GetVideoInfo(id);
        }

        return videoCache;
    }

    void DownloadVideo(int id) const override {
        if (!needReset) {
            service->DownloadVideo(id);
        }
    }

private:
    bool needReset{};
    ThirdPartyTvLib* service;
    string listCache, videoCache;
};

class TVManager {
public:
    TVManager(ThirdPartyTvLib* thirdPartyTvLib) {
        service = thirdPartyTvLib;
    }

    void RenderVideoPage(int id) {
        string info = service->GetVideoInfo(id);
        cout << info;
    }

    void RenderListPanel() {
        string list = service->ListVideos();
        cout << list;
    }

    void ReactOnUserInput(int id) {
        RenderVideoPage(id);
        RenderListPanel();
    }

protected:
    ThirdPartyTvLib* service;
};

int main(int argc, char *argv[]) {
    auto aTvService = new ThirdPartyTvClass();
    auto aTvProxy = new CachedTvClass(aTvService);
    auto manger = new TVManager(aTvProxy);

    manger->ReactOnUserInput(0);
    cout << "--------------------" << endl;
    manger->ReactOnUserInput(1);

    delete aTvProxy;
    delete aTvService;
    delete manger;

    return 0;
}