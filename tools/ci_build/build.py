import os
import sys
import argparse
import subprocess
import shutil
import logging
import multiprocessing

class BaseError(Exception):
    """Base class for errors originating from build.py."""
    pass


class BuildError(BaseError):
    """Error from running build steps."""

    def __init__(self, *messages):
        super().__init__("\n".join(messages))


class UsageError(BaseError):
    """Usage related error."""

    def __init__(self, message):
        super().__init__(message)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
log = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--build_dir", 
            required=True, 
            help="Path to the build directory"        
    )

    parser.add_argument(
            "--config",
            required=True,
            help="Type of CMAKE_BUILD_TYPE"
    )

    parser.add_argument(
            "--skip_submodule_sync",
            action='store_true',
            help="Don't do a 'git submodule update'. Makes the Update phase faster."
    )

    parser.add_argument(
            "--cmake_path",
            default="cmake",
            help="Path to the CMake program."
    )

    parser.add_argument(
            "--protoc_path",
            default="protoc",
            help="Path to the Proto Compiler program"
    )

    parser.add_argument(
            "--skip_build_test",
            action='store_true',
            help="Turn ON to skip build unit test."
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Turn ON to parallel build"
    )

    parser.add_argument(
        "--use_sse",
        action='store_true',
        help="Turn ON to use sse instruction for processor"
    )

    parser.add_argument(
        "--use_openmp",
        action='store_true',
        help="Turn ON to use sse instruction for processor"
    )

    parser.add_argument(
            "--use_double_type",
            action='store_true',
            help="Turn ON to use double type instead of float"
    )

    parser.add_argument(
            "--use_determenistic_gen",
            action='store_true',
            help="Turn ON to use random gen with fixed seed"
    )

    return parser.parse_args()

def update_submodules(source_dir):
    subprocess.run(["git", "submodule", "sync", "--recursive"])
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"])

def generate_build_tree(cmake_path, source_dir, build_dir, args):
    cmake_dir = os.path.join(source_dir, 'cmake')
    cmake_args = [
        cmake_path, "-S", cmake_dir, "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=" + args.config,
        "-Dxsdnn_BUILD_TEST=" + ("OFF" if args.skip_build_test else "ON"),
        "-Dxsdnn_USE_DOUBLE=" + ("ON" if args.use_double_type else "OFF"),
        "-Dxsdnn_USE_DETERMENISTIC_GEN=" + ("ON" if args.use_determenistic_gen else "OFF"),
        "-Dxsdnn_USE_SSE=" + ("ON" if args.use_sse else "OFF"),
        "-Dxsdnn_USE_OPENMP=" + ("ON" if args.use_openmp else "OFF"),
        ]

    return cmake_args

def generate_build_tree_mmpack(build_dir, args):
    cmake_args = [
        "./cmake/external/mmpack/build.sh",
        "--config=" + args.config
        # TODO: аргументы для билда mmpack добавлять здесь
    ]

    if args.parallel:
        cmake_args.append("--parallel")

    if args.skip_submodule_sync:
        cmake_args.append("--skip_submodule_sync")


    # Копирование libmmpack.a в директорию текущего билда
    if "Linux" in build_dir:
        os = "Linux"
    else:
        raise BuildError("Unsupported OS")

    os_args = [
        "cp", "cmake/external/mmpack/build/" + os + "/" + args.config + "/libmmpack.a", build_dir
    ]
    return cmake_args, os_args

def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    executable_path = shutil.which(command_or_path)
    if executable_path is None:
        raise BuildError("Failed to resolve executable path for "
                         "'{}'.".format(command_or_path))
    return os.path.abspath(executable_path)

def try_create_dir(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)

def compile_onnx(protoc_path, source_dir):
    protoc_path = resolve_executable_path(protoc_path)
    onnx_proto = source_dir + "/cmake/external/onnx/onnx"
    destination = source_dir + "/include/converter"
    return subprocess.run([protoc_path, f"--cpp_out={destination}", f"--proto_path={onnx_proto}", "onnx.proto3"])

def run_build(build_tree):
    return subprocess.run(build_tree)

def make(build_dir, args):
    make_args = [
        "make",
        "-C", build_dir,
        f"-j{multiprocessing.cpu_count() if args.parallel else 1}"
    ]

    return subprocess.run(make_args)

def main():
    args = parse_arguments()
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))
    
    if args.config in ['Debug', 'Release', 'RelWithDebInfo']:
        build_dir = args.build_dir + "/" + args.config
    else:
        raise UsageError("Unsupported type of config")
    
    if not args.skip_submodule_sync:
        update_submodules(source_dir)
    
    cmake_path = resolve_executable_path(args.cmake_path)
    cmake_args = generate_build_tree(cmake_path, source_dir, build_dir, args)
    try_create_dir(build_dir)
    compile_onnx(args.protoc_path, source_dir)
    run_build(cmake_args)
    # Start making
    make(build_dir, args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BaseError as e:
        log.error(str(e))
        sys.exit(1)
