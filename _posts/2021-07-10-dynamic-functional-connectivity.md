---
layout: post
title:  "Dynamic Functional Connectivity with Julia"
date:   2021-07-10 00:00:00 -0500
tags: ['coding', 'neuroscience']
minute: 5
---

At the Center for Brain Circuit Therapeutics, almost all of our computational research revolves around functional connectivity analysis of resting-state functional MRI (rsfMRI). While there are many subtleties in the preprocessing of rsfMRI and the interpretation of functional connectivity, the short version is:

1. Extract a representative timecourse for a region of interest (ROI)
2. Correlate that representative timecourse to every voxel in the brain

![connectivity diagram](/assets/connectivity.png)

With >200k voxels in a 2mm resolution brain volume and 1000 brains to process for each region of interest, this amounts to quite a lot of Pearson correlations. When I joined the lab, this was being done with a Matlab script that parallelized across ROIs, which could process approximately 1 ROI every 10 minutes. I rewrote the pipeline using Python with Numba JIT acceleration, which, with some math trickery, brought the processing time to about 30 seconds per ROI. 

While this is probably as fast as this particular analysis method is going to get without some sort of GPU optimization, I was always bothered by how much effort it took to create that Python application. For example, the only way to calculate the Pearson correlation between two vectors is to use `scipy.stats.pearsonr()`; Numpy only provides `numpy.corrcoef()` which can *only* produce whole correlation matrices. As I quickly discovered, `pearsonr()` is not a particularly performant function, partially because it calculates the p-value of the correlation in addition to the r-value, a behavior that cannot be turned off. Perhaps I'm just not a particularly competent programmer, but I also struggled to use `nunmpy.apply_along_axis()` to try to vectorize it. In the end, I had to manually compute the Pearson correlation using its dot-product form and basic Numpy methods. 

That wasn't the end of my problems; I wanted to use Numba to optimize the computationally heavy parts of the code. It turns out that the promise of being able to wrap everything in `@jit` is a bit more complicated than it seems. Certain Numpy functions aren't supported (e.g. `numpy.mean(axis=1)`). I suppose where I went wrong was that I applied the Numba decorators after the bulk of the code had been finished, which meant that I had to rethink and rewrite the unsupported parts, while what I should have done is started with the optimizations in the first place. 

All these little problems aggregated to the effect that the tools that I had created were quite inflexible. In fact, I don't think that I've since reused any of it for similar tasks other than slight modifications to the exact same functional connectivity analysis. And so, when I started reading about Julia, I began to feel the familiar, impulsive itch to jump ship to yet another different language.

[Julia](https://julialang.org/) is a dynamic, JIT compiled language geared toward scientific/numerical computing, which promises the speed of native C while using a high-level Python-like syntax. Additionally, distributed and GPU computing are built in by default with the `GPUArray` type and the `distributed` package. With lofty promises like that, it's hard to see how anyone wouldn't be intrigued at the least. 

I decided that dynamic functional connectivity analysis might be a good test case for me to try out the language. Dynamic connectivity is a technique that hasn't been well explored in our lab yet; the idea is to take sliding-window chunks out of a resting state timecourse and analyze how connectivity patterns change over the course of a scan. While this does raise the noise floor quite a bit, the hope is that it can capture slower, oscilatory patterns or trends. 

# Data Munging
"Julia sounds great and all", you say, "but Python has such a bigger community. And a quick google shows that neuroimaging specific tools for Julia are virtually nonexistent!"

All of this is true, of course, but thankfully the PyCall Julia package lets you steal all of Python's thunder: 
```julia
 nl_input_data = pyimport("nilearn.input_data")
 nl_image = pyimport("nilearn.image")
 nl_plotting = pyimport("nilearn.plotting")
```
That's right, you can use your favorite nilearn/nibabel functions directly from Julia. Check this out:
```julia
julia> masker = nl_input_data.NiftiMasker(mask).fit();

julia> roi_vec = masker.transform(roi)[1,:];

julia> typeof(roi_vec)
Array{Float64,1}
```
You can use nilearn's NiftiMasker to convert a Nifti to a vector and dump it directly into a Julia array that's ready to use. 

# Making a Dynamic Connectivity Map
In my opinion, the most fun part of using Julia is being able to recklessly write for loops, just like you did when you took Computer Science 101. To correlate a timecourse to every other timecourse in the brain mask, all you need is this:
```julia
 function correlate_wholebrain(roi_tc::Array, brain_tc::Array)
     corrmap = zeros(Float64, size(brain_tc)[2])
     for i in 1:size(brain_tc)[2]
         corr = cor(roi_tc, brain_tc[:,i])
         corrmap[i] = corr
     end
     # Fisher-Z transform
     return(atanh.(corrmap))
 end
 ```
 Coming from a Python background, you're probably wincing looking at that for loop. But in Julia, this is just as performant as a vectorized numpy ufunc. Another feature to note is `atanh.(corrmap)`; the `.` operator applies any function element-wise to an array.
 
Applying this for sliding window connectivity with a window size of 10, an increment of 1, and a brain mask of 292,019 voxels:

```julia
julia> include("generate_conn.jl")
Calculating window #1 out of 100
  0.684962 seconds (292.03 k allocations: 51.243 MiB, 9.62% gc time)
Calculating window #2 out of 100
  0.666912 seconds (292.03 k allocations: 51.243 MiB, 7.77% gc time)
Calculating window #3 out of 100
  0.611829 seconds (292.03 k allocations: 51.243 MiB, 0.34% gc time)
Calculating window #4 out of 100
  0.610506 seconds (292.03 k allocations: 51.243 MiB, 0.23% gc time)
Calculating window #5 out of 100
  0.610400 seconds (292.03 k allocations: 51.243 MiB, 0.24% gc time)
Calculating window #6 out of 100
  0.609987 seconds (292.03 k allocations: 51.243 MiB, 0.23% gc time)
Calculating window #7 out of 100
  0.609796 seconds (292.03 k allocations: 51.243 MiB, 0.23% gc time)
```

Not bad at all! 

Now let's try running it with an average timecourse from a prosopagnosia-causing lesion (segmented previously for a [very cool paper](https://doi.org/10.1093/brain/awz332)). We can use `nilearn.plotting` to output a jpg of each map, and then we can use ffmpeg to concatenate them all into a gif. 

![animation](/assets/animation.gif)

(Please excuse the weird ffmpeg artifacts. I'm too lazy to track down the cause.) 

Of course, there is still much to do before the data generated is scientifically useful in any way. However, the ease with which I was able to throw together a performant piece of code was very cool. All of the computational heavy lifting was done with built-in Julia code, only using external packages to work with Niftis and plot brain images. I can certainly see myself using Julia for neuroimaging in the future, especially if I'm prototyping some nonstandard analysis that isn't already built into FSL/SPM/nilearn.
