// fgx_dump: print the canonical manifest of a .fgx file (docs/FGX_FORMAT.md §7).
//   fgx_dump model.fgx [manifest.txt]
// Exit codes: 0 ok, 1 usage, 2 rejected file.
#include "forgert/import/fgx.h"

#include <cstdio>
#include <iostream>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: fgx_dump model.fgx [manifest.txt]\n";
        return 1;
    }
    std::string text;
    try {
        text = forgert::fgx::dump(forgert::fgx::readFile(argv[1]));
    } catch (const forgert::fgx::FormatError& e) {
        std::cerr << "fgx_dump: " << argv[1] << ": " << e.what() << "\n";
        return 2;
    }
    // Write bytes exactly; no CRLF translation.
    std::FILE* out = stdout;
    if (argc == 3) {
        out = std::fopen(argv[2], "wb");
        if (!out) { std::cerr << "fgx_dump: cannot write " << argv[2] << "\n"; return 1; }
    } else {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
    }
    const bool ok = std::fwrite(text.data(), 1, text.size(), out) == text.size();
    if (out != stdout) std::fclose(out);
    return ok ? 0 : 1;
}