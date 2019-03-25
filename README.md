# face_matching
# Face Matching Repository

## Extract features with any of the following:
- vggface_feature_extraction
- facenet_feature_extraction
- lbp_feature_extraction

### Example usage:
~~~bash
python3 vggface_feature_extraction.py -n resnet50 -s ../image_list.txt -d ../output_folder
~~~

The default weights for VGGFace and ResNet50 are trained on VGGFace and VGGFace2, respectively.

## Match with:
- mult_feature_match_list

### Example usage:
~~~bash
python3 mult_feature_match_list.py -p ../probe_list.txt -o ../output_results/ -d MORPH -gr AA -m 1
~~~
## Plot ROC, FMR/FNMR, and histograms using the plot functions inside the plot folder.

### Example usage:
~~~bash
python3 plot_relative_freq_histogram.py -a1 ../authentic_dist1.txt -i1 ../impostor_dist1.txt -l1 Label1 -a2 ../authentic_dist2.txt -i2 ../impostor_dist2.txt -l2 Label2 -t 'Tittle' -d ../save_folder -n output
~~~

## Related papers:

O. M. Parkhi, A. Vedaldi, and A. Zisserman. Deep face recognition. In BMVC, 2015
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385,
2015.
Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman. Vggface2: A dataset for recognising faces across pose and age. In Face and Gesture Recognition, 2018.
Y. Guo, L. Zhang, Y. Hu, X. He, and J. Gao. Ms-celeb-1m: A dataset and benchmark for large-scale face recognition. In European Conference on Computer Vision, 2016.
X. Tan and B. Triggs. Enhanced local texture feature sets for face recognition under difficult lighting conditions. IEEE Transactions on Image Processing, 2010.




