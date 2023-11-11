
        message(WARNING "Using `OGLCompilerTargets.cmake` is deprecated: use `find_package(glslang)` to find glslang CMake targets.")

        if (NOT TARGET glslang::OGLCompiler)
            include("/home/jakeluo/Documents/Hsuanwu/deployment/ncnn/ncnn/build/install/lib/cmake/glslang/glslang-targets.cmake")
        endif()

        add_library(OGLCompiler ALIAS glslang::OGLCompiler)
    