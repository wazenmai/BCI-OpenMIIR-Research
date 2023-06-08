# BCI Final - Analyze EEG Taken During Music Perception and Imagination
111062610 陳奕君

## Usage
### Requirements
Install basic library.
```
pip3 install -r requirements.txt
```
Download openmiir information and asrpy
```
git clone https://github.com/sstober/openmiir.git
git clone https://github.com/DiGyt/asrpy.git
cd asrpy
pip3 install -e .
```
In asrpy.asrpy, change `np.int` to `np.int64`.

## Reference
Paper
- Stober, S., Sternin, A., Owen, A.M., & Grahn, J.A. (2015). Towards Music Imagery Information Retrieval: Introducing the OpenMIIR Dataset of EEG Recordings from Music Perception and Imagination. International Society for Music Information Retrieval Conference.
- Stober, S. (2017). Toward Studying Music Cognition with Information Retrieval Techniques: Lessons Learned from the OpenMIIR Initiative. Frontiers in Psychology, 8.
- Ntalampiras, S., & Potamitis, I. (2019). A Statistical Inference Framework for Understanding Music-Related Brain Activity. IEEE Journal of Selected Topics in Signal Processing, 13, 275-284.
- S. Ntalampiras, "Unsupervised Spectral Clustering of Music-Related Brain Activity," 2019 15th International Conference on Signal-Image Technology & Internet-Based Systems (SITIS), Sorrento, Italy, 2019, pp. 193-197, doi: 10.1109/SITIS.2019.00041.

Website
- OpenMIIR github: https://github.com/sstober/openmiir
- Deepthought github: https://github.com/sstober/deepthought
