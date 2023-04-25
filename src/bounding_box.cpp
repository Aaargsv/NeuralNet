#include "bounding_box.h"

float compute_segments_overlap_length(float l0, float r0, float l1, float r1)
{
    float l2 = (l1 > l0) ? l1 : l0;
    float r2 = (r1 < r0) ? r1 : r0;
    float overlap_length = r2 - l2;
    if (overlap_length < 0)
        return 0;
    else
        return overlap_length;
}

float compute_intersection_area(float x0, float y0, float w0, float h0,
                                float x1, float y1, float w1, float h1)
{
    float left0   = x0 - w0 / 2;
    float right0  = x0 + w0 / 2;
    float left1   = x1 - w1 / 2;
    float right1  = x1 + w1 / 2;
    float overlap_length_x = compute_segments_overlap_length(left0, right0, left1, right1);

    float top0    = y0 - h0 / 2;
    float bottom0 = y0 + h0 / 2;
    float top1    = y1 - h1 / 2;
    float bottom1 = y1 + h1 / 2;
    float overlap_length_y = compute_segments_overlap_length(top0, bottom0, top1, bottom1);

    return overlap_length_x * overlap_length_y;
}

float compute_IoU(BoundingBox &bb0, BoundingBox &bb1)
{

    float x0, y0, w0, h0;
    float x1, y1, w1, h1;

    x0 = bb0.bx;
    y0 = bb0.by;
    w0 = bb0.bw;
    h0 = bb0.bh;

    x1 = bb1.bx;
    y1 = bb1.by;
    w1 = bb1.bw;
    h1 = bb1.bh;

    float intersection_area = compute_intersection_area(x0, y0, w0, h0,
                                                        x1, y1, w1, h1);
    float union_area = w0 * h0 + w1 * h0 - intersection_area;
    return intersection_area / union_area;
}