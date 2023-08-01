# Contrastive Multi-view Kernel Learning

Matlab implementation for TPAMI23 paper:

- Jiyuan Liu, Xinwang Liu\*, Yuexiang Yang, Qing Liao, Yuanqing Xia: [Contrastive Multi-view Kernel Learning](https://liujiyuan13.github.io/pubs/cmkl_published.pdf). IEEE Transactions on Pattern Analysis and Machine Intelligence, TPAMI, 2023.

## Introduction
**Abstract**

Kernel method is a proven technique in multi-view learning. 
It implicitly defines a Hilbert space where samples can be linearly separated. 
Most kernel-based multi-view learning algorithms compute a kernel function aggregating and compressing the views into a single kernel.
However, existing approaches compute the kernels independently for each view. This ignores complementary information across views and thus may result in a bad kernel choice. 
In contrast, we propose the Contrastive Multi-view Kernel — a novel kernel function based on the emerging contrastive learning framework.
The Contrastive Multi-view Kernel implicitly embeds the views into a joint semantic space where all of them resemble each other while promoting to learn diverse views. 
We validate the method’s effectiveness in a large empirical study. 
It is worth noting that the proposed kernel functions share the types and parameters with traditional ones, making them fully compatible with existing kernel theory and application. 
On this basis, we also propose a contrastive multi-view clustering framework and instantiate it with multiple kernel k-means, achieving a promising performance. 
To the best of our knowledge, this is the first attempt to explore kernel generation in multi-view setting and the first approach to
use contrastive learning for a multi-view kernel learning.

## Code structure

```
...
+ data				# dataset folder
+ eval 				# Matlab functions for evaluation
+ save 				# result folder
+ tool 				# tool functions
.gitignore 			
CMK.py 				# main file of CMK
CMK_batch.py 		# main file of CMK in batch training (for large-scale data)
env_torch.yaml  	# environment details of Pytorch
LICENSE.py 			# license file
README.md 
run.py 			    # run file (example) of CMK.py
run_batch.py 		# run file (example) of CMK_batch.py
test.m 				# test file of CMK
test_nystrom.py 	# test file of CMK in batch training (for large-scale data)
```

## Usage

1. Clone to the local.
```
> git clone https://github.com/liujiyuan13/CMK-code_release.git CMK-code_release
```
2. Install required packages.
```
> conda env create -f env_torch.yml
```
3. Train CMK and CMKKM. 
```
> python run.py			# for small datasets, such as BBC (CMK and CMKKM)
or
> python run_batch.py	# for large-scale datasets, such as YtVideo (CMK)
```
4. Test CMK and CMKKM.
```
> test.m	    	# corresponding to run.py
or
> test_batch.py		# corresponding to run_batch.py
```


## Citation

If you find our code useful, please cite:

	@article{DBLP:journals/pami/LiuLYLX23,
	  author       = {Jiyuan Liu and
	                  Xinwang Liu and
	                  Yuexiang Yang and
	                  Qing Liao and
	                  Yuanqing Xia},
	  title        = {Contrastive Multi-View Kernel Learning},
	  journal      = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
	  volume       = {45},
	  number       = {8},
	  pages        = {9552--9566},
	  year         = {2023},
	  url          = {https://doi.org/10.1109/TPAMI.2023.3253211},
	  doi          = {10.1109/TPAMI.2023.3253211},
	  timestamp    = {Fri, 21 Jul 2023 22:26:14 +0200},
	  biburl       = {https://dblp.org/rec/journals/pami/LiuLYLX23.bib},
	  bibsource    = {dblp computer science bibliography, https://dblp.org}
	}

## Licence

This repository is under [GPL V3](https://github.com/liujiyuan13/CMK-code_release/blob/main/LICENSE).

## More
- For more related researches, please visit my homepage: [https://liujiyuan13.github.io](https://liujiyuan13.github.io).
- For data and discussion, please message me: liujiyuan13@nudt.edu.cn