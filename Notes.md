Profiling pyCUDA via the CUDA Visual Profiler (when using in a conda environment)

I have configured my python code within a conda environment so the execute it using Python requires that the environment is activated. The same would be true of virtualenv if I had used this rather than conda. As I am on a windows machine I have created a windows batch file to configure and then execute my PyCUDA script. For example my conda environment (called GPy) is activated and my CUDA code executes.

	activate GPy
	SET CUDA_DEVICE=2
	python my_python_file.py
	deactivate GPy
	
The ```CUDA_DEVICE``` environment variable ensure that the correct GPU is used. In my case device id of 2 is my Tesla K40. Running my particular script which has heavy use of double precision is requires the K40 rather than either of my Titan X GPUs (with device ids 0,1). To check the device id in your own system you can check the NVIDIA settings control panel.

The above is enough to create a batch file to run your code however for profiling there is an additional requirement. The code needs to be modified to flush the GPU buffers. This can be done by calling the following pycuda function.

	pycuda.driver.stop_profiler()
	
Alternatively if you want to only profile part of your application you can use the following functions within your python script.

	pycuda.driver.start_profiler()
	
	... your code
	
	pycuda.driver.stop_profiler()
	
You can now execute your batch file within the visual profiler as if it were a standalone GPU executable.

Now to do some optimisation...
