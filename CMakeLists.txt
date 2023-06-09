# cmake needs this line
cmake_minimum_required(VERSION 3.17)
# Enable C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Build directory" FORCE)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Build directory" FORCE)
if(NOT CMAKE_BUILD_TYPE AND NOT (MSVC_IDE OR XCODE))
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type, one of: Release, Debug"
            FORCE)
endif()
# Set target name, change with your favorite name
project(main) #Keep it the same as executable name. E.g., if its main.cpp, keep it main
message( STATUS "Build type: ${CMAKE_BUILD_TYPE}" )
message( STATUS "Configuration types: ${CMAKE_CONFIGURATION_TYPES}" )
# Detect 64 or 32 bit
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64-bit project
    SET(64_BIT_OS TRUE)
    message( STATUS "A 64-bit OS detected" )
else()
    SET(64_BIT_OS FALSE)
    message( STATUS "A 32-bit OS detected" )
endif()
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
if(WIN32)
    set(OpenCV_DIR "C:/opencv/build/x64/vc15/lib")
endif()
if(APPLE)
    set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/*/include/opencv4")
endif()
find_package(OpenCV REQUIRED)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS " config: ${OpenCV_DIR}")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")
file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h)
include_directories("${CMAKE_CURRENT_LIST_DIR}")
# Since there are a lot of examples I'm going to use a macro to simplify this
# CMakeLists.txt file. However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
macro(add_example name header1 header2 header3 header4 header5)
    add_executable(Mask-Detection ${name}.cpp ${header1}.h ${header2}.h ${header3}.h ${header4}.h ${header5}.h)
    target_link_libraries(Mask-Detection ${OpenCV_LIBS})
endmacro()
# if an example requires GUI, call this macro to check DLIB_NO_GUI_SUPPORT to include or exclude
macro(add_gui_example name)
    if (DLIB_NO_GUI_SUPPORT)
        message("No GUI support, so we won't build the ${name} example.")
    else()
        add_example(${name})
    endif()
endmacro()
add_example(main headers/helper headers/preprocessing headers/facedetection headers/postprocessing headers/maskdetection) #Give the executable name without the cpp. E.g, if its main.cpp, give main