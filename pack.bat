@echo off

ECHO.
ECHO =======================================================
ECHO             开始构建排序算法可视化工具...
ECHO =======================================================
ECHO.

SET output_folder=BuildOutput

pyinstaller --name "SortVisualizer" ^
            --onefile ^
            --windowed ^
            --add-data "%~dp0theme.json;." ^
            --distpath "%output_folder%/dist" ^
            --workpath "%output_folder%/build" ^
            --specpath "%output_folder%" ^
            main.py

ECHO.
ECHO =======================================================
ECHO                   构建完成！
ECHO.
ECHO - 最终的 .exe 文件位于: %output_folder%\dist
ECHO - 按任意键退出...
ECHO =======================================================
ECHO.

pause >nul
