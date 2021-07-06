---
layout: post
title:  "Dynamic Functional Connectivity with Julia"
date:   2021-01-01 00:00:00 -0500
tags: coding
minute: 15
---

At the Center for Brain Circuit Therapeutics, almost all of our computational research revolves around functional connectivity analysis of resting-state functional MRI (rsfMRI). While there are many subtleties in the preprocessing of rsfMRI and the interpretation of functional connectivity, but the short version of our primary analysis is:

1. Extract a representative timecourse for the region of interest (ROI)
2. Correlate that representative timecourse to every voxel in the brain 

![connectivity diagram](/assets/connectivity.png)

With >200k voxels in a 2mm resolution brain volume and 1000 brains to process for each region of interest, this amounts to quite a lot of Pearson correlations. When I joined the lab, this was being done with a Matlab script that parallelized across ROIs, which could process approximately 1 ROI every 10 minutes. I rewrote the pipeline using Python with Numba JIT acceleration, which, with some math trickery, brought the processing time to about 30 seconds per ROI. 

While this is probably as fast as this particular analysis method is going to get without some sort of GPU optimization, I was always bothered by how much effort it took to create that Python application. For example, the only way to calculate the Pearson correlation between two vectors is to use `scipy.stats.pearsonr()`; Numpy only provides `numpy.corrcoef()` which can *only* produce whole correlation matrices. As I quickly discovered, `pearsonr()` is not a particularly performant function, partially because it calculates the p-value of the correlation in addition to the r-value, a behavior that cannot be turned off. Perhaps I'm just not a particularly competent programmer, but I also struggled to use `nunmpy.apply_along_axis()` to try to vectorize it. In the end, I had to manually compute the Pearson correlation using its dot-product form and basic Numpy methods. 

That wasn't the end of my problems; I wanted to use Numba to optimize the computationally heavy parts of the code. It turns out that the promise of being able to wrap everything in `@jit` is a bit more complicated than it seems. Certain Numpy functions aren't supported (e.g. `numpy.mean(axis=1)`). I suppose where I went wrong was that I applied the Numba decorators after the bulk of the code had been finished, which meant that I had to rethink and rewrite the unsupported parts, while what I should have done is started with the optimizations in the first place. 

All these little problems aggregated to the effect that the tools that I had created were quite inflexible. In fact, I don't think that I've since reused any of it for similar tasks other than slight modifications to the exact same functional connectivity analysis. And so, when I started reading about Julia, I began to feel the familiar, impulsive itch to jump shit to yet another different language.

[Julia](https://julialang.org/) is a dynamic, JIT compiled language geared toward scientific/numerical computing, which promises the speed of native C while using a high-level Python-like syntax. Additionally, distributed and GPU computing are built in by default with the `GPUArray` type and the `distributed` package. With lofty promises like that, it's hard to see how anyone wouldn't be intrigued at the least. 

I decided that dynamic functional connectivity analysis might be a good test case for me to try out the language. Dynamic connectivity is a technique that hasn't been well explored in our lab yet; the idea is to take sliding-window chunks out of a resting state timecourse and analyze how connectivity patterns change over the course of a scan. While this does raise the noise floor quite a bit, the hope is that it can capture slower, oscilatory patterns or trends. 

