@echo off

IF "%VCPKG_ROOT%"=="" ECHO VCPKG_ROOT is NOT defined.
set vcpkg="-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake"
set cmake="C:\Program Files\CMake\bin\cmake.exe"
set msvc=-DMSVC=TRUE
set type=-DCMAKE_BUILD_TYPE=Release
 
cd AREC3D
cd build
%cmake% %vcpkg% %msvc% %type% ../
%cmake% --build . --config Release
cd ..
cd ..
