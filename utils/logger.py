"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    """
    键值对写入器的基类，定义了写入键值对的抽象方法。
    """
    def writekvs(self, kvs):
        """
        写入键值对的抽象方法，具体实现由子类完成。

        Args:
            kvs (dict): 要写入的键值对字典。
        """
        raise NotImplementedError


class SeqWriter(object):
    """
    序列写入器的基类，定义了写入序列的抽象方法。
    """
    def writeseq(self, seq):
        """
        写入序列的抽象方法，具体实现由子类完成。

        Args:
            seq (iterable): 要写入的序列。
        """
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    """
    以人类可读的格式将日志信息写入文件或标准输出。
    """
    def __init__(self, filename_or_file):
        """
        初始化 HumanOutputFormat 类。

        Args:
            filename_or_file (str or file object): 文件名或文件对象。
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        """
        将键值对以人类可读的表格形式写入文件。

        参数:
            kvs (dict): 要写入的键值对字典。
        """
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        """
        截断字符串，如果字符串长度超过最大长度，则在末尾添加省略号。

        参数:
            s (str): 要截断的字符串。

        返回:
            str: 截断后的字符串。
        """
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        """
        将序列元素写入文件，元素之间用空格分隔，最后换行。

        参数:
            seq (iterable): 要写入的序列。
        """
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        """
        关闭文件（如果是自己打开的文件）。
        """
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    """
    将日志信息以 JSON 格式写入文件。
    """
    def __init__(self, filename):
        """
        初始化 JSONOutputFormat 类。

        参数:
            filename (str): 要写入的文件名。
        """
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        """
        将键值对以 JSON 格式写入文件。

        参数:
            kvs (dict): 要写入的键值对字典。
        """
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        """
        关闭文件。
        """
        self.file.close()


class CSVOutputFormat(KVWriter):
    """
    将日志信息以 CSV 格式写入文件。
    """
    def __init__(self, filename):
        """
        初始化 CSVOutputFormat 类。

        参数:
            filename (str): 要写入的文件名。
        """
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        """
        将键值对以 CSV 格式写入文件。如果有新的键，会更新文件的表头。

        参数:
            kvs (dict): 要写入的键值对字典。
        """
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        """
        关闭文件。
        """
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        """
        初始化 TensorBoardOutputFormat 类。

        参数:
            dir (str): 要写入的目录。
        """
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        """
        将键值对以 TensorBoard 的数值格式写入文件。

        参数:
            kvs (dict): 要写入的键值对字典。
        """
        def summary_val(k, v):
            """
            创建 TensorBoard 摘要值。

            参数:
                k (str): 键。
                v (float): 值。

            返回:
                tf.Summary.Value: TensorBoard 摘要值。
            """
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = (
            self.step
        )  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        """
        关闭写入器。
        """
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    """
    根据指定的格式创建相应的输出格式对象。

    参数:
        format (str): 输出格式，可选值有 "stdout", "log", "json", "csv", "tensorboard"。
        ev_dir (str): 日志目录。
        log_suffix (str, optional): 日志文件后缀。

    返回:
        KVWriter or SeqWriter: 相应的输出格式对象。
    """
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def set_comm(comm):
    get_current().set_comm(comm)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


def get_current():
    """
    获取当前使用的日志记录器。如果当前日志记录器为 None，则配置默认日志记录器。

    返回:
        Logger: 当前使用的日志记录器。
    """
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    """
    日志记录器类，负责管理日志的记录和输出。
    """
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        """
        初始化 Logger 类。

        参数:
            dir (str): 日志目录。
            output_formats (list): 输出格式对象列表。
            comm (optional): 通信对象。
        """
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        """
        记录一个键值对。

        参数:
            key (str): 键。
            val: 值。
        """
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        """
        记录一个键值对，并对多次记录的值进行平均。

        参数:
            key (str): 键。
            val: 值。
        """
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        """
        写入当前迭代的所有键值对。

        返回:
            dict: 写入的键值对字典。
        """
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        """
        根据日志级别写入日志信息。
        只有当日志级别大于等于当前设置的日志级别时，才会写入日志。

        参数:
            *args: 要写入的日志信息。
            level (int, optional): 日志级别，默认为 INFO。
        """
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        """
        设置当前日志记录器的日志级别。

        参数:
            level (int): 日志级别。
        """
        self.level = level

    def set_comm(self, comm):
        """
        设置当前日志记录器的通信对象。

        参数:
            comm: 通信对象。
        """
        self.comm = comm

    def get_dir(self):
        """
        获取日志文件的存储目录。

        返回:
            str: 日志文件的存储目录。
        """
        return self.dir

    def close(self):
        """
        关闭所有输出格式对象，释放资源。
        """
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        """
        将日志信息写入到所有支持序列写入的输出格式对象中。

        参数:
            args (tuple): 要写入的日志信息。
        """
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def get_rank_without_mpi_import():
    """
    获取当前进程的排名，而不导入 mpi4py 模块。
    通过检查环境变量 "PMI_RANK" 和 "OMPI_COMM_WORLD_RANK" 来确定排名。

    返回:
        int: 当前进程的排名，如果未找到相关环境变量，则返回 0。
    """
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0


def mpi_weighted_mean(comm, local_name2valcount):
    """
    Copied from: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(
                            "WARNING: tried to compute mean on non-float {}={}".format(
                                name, val
                            )
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


def configure(dir=None, format_strs=None, comm=None, log_suffix=""):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % dir)


def _configure_default_logger():
    """
    配置默认的日志记录器。
    调用 configure 函数进行配置，并将配置好的日志记录器设置为默认日志记录器。

    返回:
        None
    """
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    """
    重置日志记录器。
    如果当前日志记录器不是默认日志记录器，则关闭当前日志记录器，并将其重置为默认日志记录器。

    返回:
        None
    """
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    """
    上下文管理器，用于临时配置日志记录器。
    在上下文管理器的作用域内，使用新的配置进行日志记录；离开作用域后，恢复之前的日志记录器配置。

    参数:
        dir (str, optional): 日志文件的存储目录。
        format_strs (list, optional): 输出格式字符串列表，用于指定日志的输出方式。
        comm (optional): 通信对象，用于在分布式环境中进行数据同步。

    返回:
        None
    """
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger

