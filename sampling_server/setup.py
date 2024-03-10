from setuptools import setup, Extension
import pybind11

# 定义额外的链接器参数
extra_link_args = ['-L build/lib', '-lserver']  # 假设动态库不在标准路径
custom_include_path = 'src/engine/'

sampling_module = Extension(
    'sampling_server',  # 模块名
    sources=['sampling_server.cpp'],
    include_dirs=[
        custom_include_path,
        pybind11.get_include()  # 添加pybind11的头文件路径
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],  # 添加C++编译参数
    extra_link_args=extra_link_args,  # 添加链接器参数
)

setup(
    name='sampling_server',
    version='1.0',
    description='Python package with C++ extension for Sampling Server',
    ext_modules=[sampling_module],
)

