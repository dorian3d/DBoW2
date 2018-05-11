/*
 * File: DBoW2.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: Generic include file for the DBoW2 classes and
 *   the specialized vocabularies and databases
 * License: see the LICENSE.txt file
 *
 */

/*! \mainpage DBoW2 Library
 *
 * DBoW2 library for C++:
 * Bag-of-word image database for image retrieval.
 *
 * Written by Dorian Galvez-Lopez,
 * University of Zaragoza
 * 
 * Check my website to obtain updates: http://doriangalvez.com
 *
 * \section requirements Requirements
 * This library requires the DUtils, DUtilsCV, DVision and OpenCV libraries,
 * as well as the boost::dynamic_bitset class.
 *
 * \section citation Citation
 * If you use this software in academic works, please cite:
 <pre>
   @@ARTICLE{GalvezTRO12,
    author={Galvez-Lopez, Dorian and Tardos, J. D.}, 
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
 </pre>
 *
 */

#ifndef __D_T_DBOW2__
#define __D_T_DBOW2__

/// Includes all the data structures to manage vocabularies and image databases
namespace DBoW2
{
}

#include "TemplatedVocabulary_ORB_SLAM.h"
#include "TemplatedDatabase.h"
#include "BowVector.h"
#include "FeatureVector.h"
#include "QueryResults.h"
#include "FBrief.h"
#include "FORB.h"
#include "FSurf64.h"
#include "FCNN.h"

/// ORB Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> 
  OrbVocabulary;

/// FORB Database
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> 
  OrbDatabase;
  
/// BRIEF Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FBrief::TDescriptor, DBoW2::FBrief> 
  BriefVocabulary;

/// BRIEF Database
typedef DBoW2::TemplatedDatabase<DBoW2::FBrief::TDescriptor, DBoW2::FBrief> 
  BriefDatabase;

/// SURF Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FSurf64::TDescriptor, DBoW2::FSurf64> 
  SURF64Vocabulary;

/// SURF Database
typedef DBoW2::TemplatedDatabase<DBoW2::FSurf64::TDescriptor, DBoW2::FSurf64> 
  SURF64Database;

/// CNN-feature Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FCNN::TDescriptor, DBoW2::FCNN>
  CNNVocabulary;

#endif