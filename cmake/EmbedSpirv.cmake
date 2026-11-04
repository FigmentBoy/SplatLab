# Converts a SPIR-V binary into a C++ header containing a byte array.
# Usage: cmake -DINPUT=<file.spv> -DOUTPUT=<file.h> -DNAME=<symbol> -P EmbedSpirv.cmake
file(READ "${INPUT}" SPV_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," SPV_BYTES "${SPV_HEX}")
file(WRITE "${OUTPUT}"
    "// Generated from ${INPUT}. Do not edit.\n"
    "#pragma once\n"
    "#include <cstddef>\n"
    "alignas(4) inline constexpr unsigned char ${NAME}[] = {${SPV_BYTES}};\n"
    "inline constexpr std::size_t ${NAME}_size = sizeof(${NAME});\n")
