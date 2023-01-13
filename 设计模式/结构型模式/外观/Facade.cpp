#include <iostream>

using namespace std;

class VideoFile {
public:
    explicit VideoFile(const string& name) { filename = name; }
    VideoFile(const string& buffer, string format) {
        this->filename = buffer;
        this->format = format;
    }

    void Save() {
        cout << filename << "." << format << " saved." << endl;
    };

private:
    string filename;
    string format;
};

class CodecFactory {
public:
    CodecFactory() = default;

    static CodecFactory* Extract(VideoFile* videoFile) {
        cout << "Extract codec type" << endl;
        return new CodecFactory();
    }
};

class OggCompressionCodec : public CodecFactory {};

class MPEG4CompressionCodec : public CodecFactory {};

class BitrateReader {
public:
    static string Read(string name, CodecFactory* codecFactory) {
        cout << "Bitrate reader reading " << name << endl;
        return name;
    }

    static string Convert(string buffer, CodecFactory* codecFactory) {
        cout << "Bitrate reader converting " << buffer << endl;
        return buffer;
    }
};

class AudioMixer {
public:
    static string Fix(string audio) {
        cout << "Audio mixer fixed" << endl;
        return audio;
    };
};

class VideoConverter {
public:
    VideoFile* Convert(string filename, string format) {
        auto file = new VideoFile(filename);
        auto codec = CodecFactory::Extract(file);

        CodecFactory* formatCodec = nullptr;
        if (format == "mp4") {
            formatCodec = new MPEG4CompressionCodec();
        } else {
            formatCodec = new OggCompressionCodec();
        }

        auto buffer = BitrateReader::Read(filename, codec);
        auto result = BitrateReader::Convert(buffer, formatCodec);
        result = AudioMixer::Fix(result);

        delete file;
        delete codec;
        delete formatCodec;

        return new VideoFile(result, format);
    }
};



int main(int argc, char *argv[]) {
    auto convertor = new VideoConverter();
    auto mp4 = convertor->Convert("funny-cats-video.ogg", "mp4");
    mp4->Save();

    return 0;
}