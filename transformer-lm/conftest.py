# 根级 conftest.py —— Windows 兼容桩。
# tests/test_tokenizer.py 顶层 `import resource`,而 resource 是 Unix 专属模块。
# 用 resource 的测试(内存限制)本身已 @skipif(not linux) 跳过,但模块级 import 会让
# Windows 收集直接失败。此文件在 win32 注入一个空桩,让测试模块能被导入。
# 对 Linux/macOS 无任何影响。
import sys

if sys.platform == "win32":
    import types

    _resource = types.ModuleType("resource")
    _resource.RLIMIT_AS = 0

    def _getrlimit(which):
        return (0, -1)

    def _setrlimit(which, limits):
        pass

    _resource.getrlimit = _getrlimit
    _resource.setrlimit = _setrlimit
    sys.modules["resource"] = _resource
