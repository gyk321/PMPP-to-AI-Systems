# 一键运行 vecMul.py：先加载 MSVC 环境（cl.exe），再用 uv 运行
# 用法：在 CH02 目录下执行 .\run.ps1
$ErrorActionPreference = "Stop"

$vs = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$dir = $PSScriptRoot  # run.ps1 所在目录

cmd /c "`"$vs`" >nul 2>&1 && cd /d `"$dir`" && uv run python vecMul.py"
exit $LASTEXITCODE
