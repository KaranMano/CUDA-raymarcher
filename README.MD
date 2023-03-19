# Volume Renderer
A volume renderer made using CUDA.

## Building the project
To generate the project makefiles run the following commands from the root directory: 

    mkdir build  
    cd build  
    cmake .. -A x64  

Since CUDA only supports x64 we have to tell cmake to build for x64. After generating the makefiles depending upon the generator used you can either use `make` to build the project using make or you can open the volume-renderer.sln file in Visual Studio and build the volume-renderer project.
