#include <string>
#include <variant>
#include <map>
#include <functional>

namespace p2c {
// using AllowedKernelArgTypes = std::variant<int, float, bool, std::string>;
using KernelArgDict = std::map<std::string, int>;

class Backend {
public:
    Backend() { optimization = false; }
    virtual ~Backend() {}
    void emitIR(const std::string &code, const std::string filename);
    void jitCompile(const std::string &code, const std::string &filename, const std::string &funcname);
    int invoke(const std::string &funcName, int args);
    bool lookup(const std::string &funcName);
    void setOptimization(bool opt) { optimization = opt; }
private:
    bool optimization;
    std::map<std::string, std::function<int(int)>> funcMap;
};
} // namesapce p2c