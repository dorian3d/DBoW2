TDBoW
=====

TDBoW is an forked version of the DBow2 library, an open source C++ library for indexing and converting
pointcloud (or image) into a bag-of-word representation.
It implements a hierarchical tree for approximating nearest neighbours in the feature space and creating
a feature vocabulary. TDBoW also implements an image database with inverted and direct files to index
pointclouds (or images) and enabling quick queries and feature comparisons.
The main differences with the previous DBow2 library are:

  * (Still work-in-progress) TDBoW support 3D descriptor calculated
  from pointcloud.
  * TDBoW classes don't need to inheritance the templated base class,
  since TDBoW using Eigen for common represent.
  * TDBoW won't compile with OpenCV any more, although it can still
  easily used based on `CVBridge.h`.
  * TDBoW now support transform descriptors in multiple threads, but
  this won't perform better when in few descriptors. Multiple threads
  query for different is support. 
  * TDBoW add binary I/O support, and compress the vocabulary using
  QuickLZ algorithm in default when binary output. On the other hand,
  it can still use YAML format for friendly reading. Specially, the
  DBoW2 vocabulary is supported for reading.
  * TDBoW using k-means Ⅱ replaced k-means++.
  * Some pieces of code have been rewritten to optimize speed.
  The interface of TDBoW has been simplified. TDBoW do not support
  compile version less than c++11.
  * Stop words won't lose IDF information any more.

TDBoW requires Eigen, yaml-cpp and Boost-filesystem.

(Still work-in-progress) TDBoW, along with DLoopDetector, has been tested on several real datasets,
yielding an execution time of 3 ms to convert the BRIEF features of an image into a bag-of-words vector
and 5 ms to look for image matches in a database with more than 19000 images.

## Citing

If you use this software in an academic work, please cite:

    @ARTICLE{Still work-in-progress}

    @ARTICLE{GalvezTRO12,
      author={G\'alvez-L\'opez, Dorian and Tard\'os, J. D.},
      journal={IEEE Transactions on Robotics},
      title={Bags of Binary Words for Fast Place Recognition in Image Sequences},
      year={2012},
      month={October},
      volume={28},
      number={5},
      pages={1188--1197},
      doi={10.1109/TRO.2012.2197158},
      ISSN={1552-3098}
    }
}

## Usage notes

### Weighting and Scoring

TDBoW implements the same weighting and scoring mechanisms as DBow2.

### Save & Load

Different from DBoW2, only vocabularies can be saved to and load from disk in TDBoW. Since TDBoW support
multiple format I/O, it support automatically decide the file format from the extension.

When using yaml format, you can also load the vocabulary data from file opened with a `cv::FileStorage`
structure.

You can save the vocabulary with any file extension. If you using binary(default), the data will default
to be compressed by QuickLZ algorithm.

## Implementation notes

### Template parameters

TDBoW has two main classes: `TemplatedVocabulary` and `TemplatedDatabase`. These implement the features
vocabulary to convert pointclouds (or images) into bag-of-words vectors and the database to index images.
These classes are templated:

    template <typename TScalar, size_t L>
    class TemplatedVocabulary {
      ...
    };

    template <class TemplatedVocabulary>
    class TemplatedDatabase {
      ...
    };

Two classes must be provided: `TScalar` is the data scalar type of the descriptor, and `L` is the
length of descriptor in `TScalar`.

For example, to work with cv::ORB descriptors, which contained by 256 bits (32 bytes). The descriptor
is designed as `Eigen::Matrix<uint8_t, 1, 32, Eigen::RowMajor>`, so we set `TScalar` to `uint8_t`, and
set `L` as `32`.

Default `meanValue`, `distance`, `toString` and `fromString` methods are already implemented, but can
still easily override by lambda methods.

### Use TDBoW in CV mode

OpenCV is a very strong library, which contains many kinds of image descriptors implement. So we add
`CVBridge.h` for OpenCV users. More details can be founded in [cv_demo](demo/cv/demo.cpp).
