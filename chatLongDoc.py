# 导入ArgumentParser用于命令行参数解析
from argparse import ArgumentParser
# 导入utils模块中的所有函数和类
from utils import *


def main():

    # 创建ArgumentParser对象，定义使用说明
    parser = ArgumentParser(
        usage="chatLongDoc.py [<args>] [-h | --help]"
    )
    # 添加`--text_path`命令行参数，用于指定文本文件的本地路径或网络URL
    parser.add_argument(
        "--text_path",
        type=str,
        default=None,
        help='''Local file path or web URL.
        e.g. 'example/example.pdf' or 'https://arxiv.org/abs/1706.03762'.
        ''',
    )
    # 添加`--memory_path`命令行参数，用于指定缓存的内存文件路径
    parser.add_argument(
        "--memory_path",
        type=str,
        default=None,
        help='''Local file path to cached memory file.
        e.g. 'memory/315b4f4c655530196ef5f4f6f1d1f0b57d24fa2b08ff006c44b535d50499b487.json'.
        If '--memory_path' is specified, the program will load the cached memory instead of processing texts from '--text_path'.
        ''',
    )

    # 解析命令行输入的参数
    args = parser.parse_args()

    # 如果没有指定文本路径和内存路径，则抛出异常
    if args.text_path is None and args.memory_path is None:
        raise ValueError(
            "Either '--text_path' or '--memory_path' must be specified.")

    chat(args.text_path)  # 使用缓存的内存文件进行聊天


if __name__ == '__main__':
    main()
