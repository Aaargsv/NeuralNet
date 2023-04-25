#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <vector>

struct BoundingBox  {
    float bx;
    float by;
    float bh;
    float bw;
    float probability;
};

/// detected bb for every class separately from each other
typedef std::vector<std::vector<BoundingBox>> BoundingBoxes;

float compute_segments_overlap_length(float l0, float r0, float l1, float r1);
float compute_intersection_area(float x0, float y0, float w0, float h0,
                                float x1, float y1, float w1, float h1);
float compute_IoU(BoundingBox &bb0, BoundingBox &bb1);



#endif //BOUNDING_BOX_H
