import os
import platform
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

# Directory containing this file
SETUP_DIR = os.path.abspath(os.path.dirname(__file__))

# Path to the parent directory (project root)
PROJECT_ROOT = os.path.abspath(os.path.join(SETUP_DIR, ".."))

class CMakeExtension(Extension):
    def __init__(self, name, source_dir=""):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DNEURONET_BUILD_PYTHON=ON'
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), ext_dir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j', str(os.cpu_count() or 2)]

        env = os.environ.copy()
        
        # Detect CUDA
        cuda_available = False
        try:
            nvcc_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
            cuda_available = True
        except:
            cuda_available = False

        # Disable CUDA if not available
        if not cuda_available:
            cmake_args += ['-DNEURONET_USE_CUDA=OFF']

        # Disable Metal on non-macOS platforms
        if platform.system() != "Darwin":
            cmake_args += ['-DNEURONET_USE_METAL=OFF']

        # Build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Execute CMake
        subprocess.check_call(['cmake', PROJECT_ROOT] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

setup(
    name="neuronet",
    version="0.1.0",
    author="Caleb Marshall",
    author_email="caleb.marshall108@gmail.com",
    description="PyTorch alternative optimized for older GPUs and macOS",
    long_description=open(os.path.join(PROJECT_ROOT, "README.md")).read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("neuronet")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
