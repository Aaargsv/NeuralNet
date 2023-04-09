#include "network.h"
#include "layers/yolo_layer.h"
#include <fstream>
#include <algorithm>

Network::Network(int height, int width,
                 const std::string &image_file_name,
                 const std::string &weight_file_name,
                 int &error_status): net_shape_(height, width, 3),
    image_file_name_(image_file_name), weights_file_name_(weight_file_name)
{
    if (load_image(image_file_name))
        error_status = 1;
    if (!error_status) {
        input_image_tensor.reserve(input_image_shape_.c * input_image_shape_.h * input_image_shape_.w);
        resized_image_tensor.reserve(net_shape_.c * net_shape_.h * net_shape_.w);
        bilinear_interpolation(input_image_tensor, input_image_shape_.h, input_image_shape_.w,
                               resized_image_tensor, net_shape_.h, net_shape_.w, net_shape_.c);
    }
}

Network::~Network()
{
    for (int i = 0; i < layers_.size(); i++) {
        delete layers_[i];
    }
}

int Network::infer()
{
    if (setup()) {
        std::cerr << "[Error]: can't setup network" << std::endl;
        return 1;
    }

    if (load_pretrained(weights_file_name_)) {
        std::cerr << "[Error]: can't load pretrained weights" << std::endl;
        return 1;
    }

    std::vector<float> *output_tensor = forward(&resized_image_tensor);
    gather_bounding_boxes();

}

std::vector<float>* Network::forward(std::vector<float>* input_image)
{
    current_tensor = input_image;
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->forward(*this);
    }
    return current_tensor;
}

int Network::setup()
{
    Shape input_layer_shape = net_shape_;
    int utility_memory_size = 0;
    int temp;
    for (int i = 0; i < layers_.size(); i++) {
        if ((temp = layers_[i]->setup(input_layer_shape, *this)) < 0) {
            return 1;
        }
        utility_memory_size = std::max(temp, utility_memory_size);
        input_layer_shape = layers_[i]->out_shape();
    }
    utility_memory.reserve(utility_memory_size);
    return 0;
}

int Network::load_image(const std::string &filename)
{
    int width, height, channels;
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cout << "[Error]: can't load image" << std::endl;
        return 1;
    }

    input_image_tensor.reserve(channels * height * width);
    input_image_shape_.reshape(height, width, channels);
    for (int c = 0; c < net_shape_.c; c++) {
        for (int h = 0; h < net_shape_.h; h++) {
            for (int w = 0; w < net_shape_.w; w++) {
                int dsr_index = c * net_shape_.w * net_shape_.h + h * net_shape_.w + w;
                int src_index = h * net_shape_.w * net_shape_.c + w * net_shape_.c + c;
                input_image_tensor[dsr_index] = static_cast<float>(data[src_index]) / 255.0f;
            }
        }
    }

    stbi_image_free(data);
    return 0;
}

/**
 * resize image via bilinear interpolation
 * @param src
 * @param src_height
 * @param src_width
 * @param dst
 * @param dst_height
 * @param dst_width
 * @param channels
 */

void Network::bilinear_interpolation(std::vector<float> &src, int src_height, int src_width,
                            std::vector<float> &dst, int dst_height, int dst_width,
                            int channels)
{
    std::vector<float> horizontal_interpolated_image(channels * src_height * dst_width);

    float h_ratio = static_cast<float>(src_height - 1) / (dst_height - 1);
    float w_ratio = static_cast<float>(src_width - 1) / (dst_width - 1);


    /// horizontal interpolation
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < src_height; h++) {
            for (int w = 0; w < dst_width; w++) {
                float brightness = 0;
                if (w == dst_width - 1 || src_width == 1) {
                    int index = c * src_height * src_width + h * src_width + (src_width - 1);
                    brightness = src[index];
                } else {
                    float x = w_ratio * w;
                    int ix = x;
                    float dx = x - ix;

                    float br0 = src[c * src_height * src_width + h * src_width + ix];
                    float br1 = src[c * src_height * src_width + h * src_width + ix + 1];
                    brightness = br0 * (1 - dx) + br1 * dx;
                }
                int index = c * src_height * dst_width + h * dst_width + w;
                horizontal_interpolated_image[index] = brightness;
            }
        }
    }

    /// vertical interpolation
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < dst_height; h++) {
            float y = h_ratio * h;
            int iy = y;
            float dy = y - iy;
            for (int w = 0; w < dst_width; w++) {
                float br0;
                int index_hii = c * src_height * dst_width + iy * dst_width + w;
                int dst_index = c * dst_height * dst_width + h * dst_width + w;
                br0 = horizontal_interpolated_image[index_hii];
                dst[dst_index] = br0 * (1 - dy);
            }

            if (h != dst_height - 1 && dst_height != 1) {
                for (int w = 0; w < dst_width; w++) {
                    float br1;
                    int index_hii = c * src_height * dst_width + (iy + 1) * dst_width + w;
                    int dst_index = c * dst_height * dst_width + h * dst_width + w;
                    br1 = horizontal_interpolated_image[index_hii];
                    dst[dst_index] += br1 * dy;
                }
            }
        }
    }
}

int Network::load_pretrained(const std::string &filename)
{
    std::ifstream file_weights(filename);
    if (!file_weights) {
        std::cout << "[Error]: can't load pretrained weights" << std::endl;
        return 1;
    }
    for (int i = 0; i < layers_.size(); i++) {
        if (layers_[i]->load_pretrained(file_weights)) {
            return 1;
        }
    }
    return 0;
}

void Network::gather_bounding_boxes()
{
    bounding_boxes_.reserve(num_classes_);
    for (int i = 0; i < layers_.size(); i++) {
        if (layers_[i]->layer_param().layer_type_ == LayerType::YOLO) {
            static_cast<YoloLayer*>(layers_[i])->get_bounding_boxes(bounding_boxes_,
                                                                    net_shape_.h, net_shape_.w);
        }
    }
}

void Network::apply_nms(float iou_threshold)
{
    for (int i = 0; i < bounding_boxes_.size(); i++) {
        std::sort(bounding_boxes_[i].begin(),
                  bounding_boxes_[i].end(),
                  [] (const BoundingBox& a, const BoundingBox& b) {
                    return a.probability > b.probability;
        });
        for (int j = 0; j < bounding_boxes_[i].size(); j++) {
            if (bounding_boxes_[i][j].probability == 0 )
                continue;
            for (int k = j + 1; k < bounding_boxes_[i].size(); k++) {
                if (bounding_boxes_[i][k].probability == 0 )
                    continue;
                if (compute_IoU(bounding_boxes_[i][j], bounding_boxes_[i][k]) > iou_threshold) {
                    bounding_boxes_[i][k].probability = 0;
                }
            }
        }
    }
}

Network& operator<<(Network &net, const Layer &layer)
{
    net.add_layer(layer.clone());
    return net;
}


