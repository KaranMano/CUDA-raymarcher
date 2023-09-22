# Volume Renderer
A simple raymarcher to render volumes.

## Building the project
To generate the project makefiles run the following commands from the root directory: 

    mkdir build  
    cd build  
    cmake .. -A x64  

Since CUDA only supports x64 we have to tell cmake to build for x64. After generating the makefiles depending upon the generator used you can either use `make` to build the project using make or you can open the volume-renderer.sln file in Visual Studio and build the volume-renderer project.

The benchmark executable can be run to time different configurations for the cpu and gpu. After execution it generates a timings.json file which can be used to create plots using a python script available in the root directory of the project.

## Results

<img src="./examples/foot.png" height="200"/>
<img src="./examples/explosion.gif" height="200"/>

## References

- [Volume Rendering for Developers: Foundations](https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/intro-volume-rendering.html)