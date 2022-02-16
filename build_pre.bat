
REM @echo off
set vcpkg="-DCMAKE_TOOLCHAIN_FILE=C:/Users/axela/Documents/GitHub/vcpkg/scripts/buildsystems/vcpkg.cmake"
set cmake="C:\Program Files\CMake\bin\cmake.exe"
set msvc=-DMSVC=TRUE
set type=-DCMAKE_BUILD_TYPE=Release
 
cd Prealign
cd build
%cmake% %vcpkg% %msvc% %type% ../
%cmake% --build . --config Release
cd ..
cd ..

