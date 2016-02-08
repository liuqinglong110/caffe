#ifndef CAFFE_DATA_TRIPLET_LAYER_HPP_
#define CAFFE_DATA_TRIPLET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype>
class DataTripletLayer : public BaseDataLayer<Dtype> {
 public:
  explicit DataTripletLayer(const LayerParameter& param): BaseDataLayer<Dtype>(param), data_param_(param.data_param()) {}
  virtual ~DataTripletLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "DataTriplet"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 3; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  shared_ptr<db::Transaction> transaction_;
  DataParameter data_param_;
  Blob<Dtype> transformed_data_;

  // class labels
  std::vector<int> labels_;
  std::vector<std::vector<int> > label_index_set_;
  
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRIPLET_LAYER_HPP_
