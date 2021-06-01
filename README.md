#  Automotive Radar Interference Mitigation data sets (ARIM)                                                                                    

We propose a novel large scale database consisting of radar data samples, generated automatically while trying to replicate a realistic automotive scenario with one source of interference.
  
We provide two ways to obtain data:
- directly by downloading the data from below listed links
- generate the data by using the provided scripts

-----------------------------------------

![map](resources/example.jpg)

-----------------------------------------                                                                                                                                      
## Download data set

https://fmiunibuc-my.sharepoint.com/:f:/g/personal/radu_ionescu_fmi_unibuc_ro/EhuJlohRmFpHswb7tizGlvsBG1SgqU12QQm0h6fHh27B6w?e=dSj4GC&fbclid=IwAR1n0-DgkUTHXxcAVQrbec-RJ2m_pND88aCunYKSIft4lag7U_czCoDUWcs

https://arxiv.org/abs/2007.11102

## Generate data set
#### In order to generate the ARIM data set:
1. Run the matlab script arim_matlab/main.m
2. Move the generated file (arim.mat) in X directory
3. Run the process.py script as follows:
```bash
python process.py --arim_data_path path/to/X/dir --output_dataset_path path/to/save
```

#### Information

After the above steps you will have in the path/to/save directory two files: **arim_train.npy** and **arim_test.npy**.
Those files contains the subsets for training (which could be split also in train and evaluation, as described in our paper) and testing.

In order to load the data in python you should run:
```python
import numpy as np
arim = np.load("path/to/dataset", allow_pickle=True)

sb_raw = arim[()]['sb'] # Data with interference
sb0_raw = arim[()]['sb0'] # Data without interference
amplitudes = arim[()]['amplitudes'] # Amplitude information for targets
```
> In order to work properly you need to have a python version older than 3.6
>> We used the following versions:
>> python 3.6.8,
>> numpy 1.17.3

## Cite us

BibTeX:

    @inproceedings{ristea2020fully,
      title={Fully convolutional neural networks for automotive radar interference mitigation},
      author={Ristea, Nicolae-C{\u{a}}t{\u{a}}lin and Anghel, Andrei and Ionescu, Radu Tudor},
        booktitle={Proceedings of VTC},
      year={2020}
    }

## You can send your questions or suggestions to: 
r.catalin196@yahoo.ro, raducu.ionescu@gmail.com

### Last Update:
June 1, 2021
