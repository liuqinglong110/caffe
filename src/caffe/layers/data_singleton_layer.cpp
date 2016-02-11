#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <stdlib.h>
#include <time.h> 

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_singleton_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataSingletonLayer<Dtype>::~DataSingletonLayer() { }

template <typename Dtype>
void DataSingletonLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->data_transformer_.reset(
      new DataTransformer<Dtype>(this->transform_param_, this->phase_));
  this->data_transformer_->InitRand();

  // open DB
  db_ = shared_ptr<db::DB>(db::GetDB(data_param_.backend()));
  db_->Open(data_param_.source(), db::WRITE);
  cursor_ = shared_ptr<db::Cursor>(db_->NewCursor());
  transaction_ = shared_ptr<db::Transaction>(db_->NewTransaction());
  size_t num_entries = cursor_->num_entries();

  // read all the class labels
  Datum datum;
  int i = 0;
  int max_label = -1;
  labels_.clear();
  labels_.resize(num_entries);
  while(cursor_->valid())
  {
    datum.ParseFromString(cursor_->value());
    cursor_->Next();
    labels_[i] = datum.label();
    if (labels_[i] > max_label)
      max_label = labels_[i];
    i++;
  }

  // contruct the label index set
  label_index_set_.clear();
  label_index_set_.resize(max_label + 1);
  for(i = 0; i <= max_label; i++)
    label_index_set_[i].clear();
  for(i = 0; i < num_entries; i++)
    label_index_set_[labels_[i]].push_back(i);

  // reset
  cursor_->SeekToFirst();

  const int batch_size = data_param_.batch_size();
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  // Reshape top[0] according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label blob
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
}

template <typename Dtype>
void DataSingletonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Datum datum;
  Blob<Dtype> data, label;
  const int batch_size = data_param_.batch_size();

  datum.ParseFromString(cursor_->value());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  data.Reshape(top_shape);

  vector<int> label_shape(1, batch_size);
  label.Reshape(label_shape);

  // read the first half
  Dtype* top_data = data.mutable_cpu_data();
  Dtype* top_label = label.mutable_cpu_data();
  int item_num = batch_size - batch_size / 2;
  for (int item_id = 0; item_id < item_num; ++item_id)
  {
    datum.ParseFromString(cursor_->value());
    top_label[item_id] = datum.label();

    // Apply data transformations (mirror, scale, crop...)
    int offset = data.offset(item_id);
    transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(transformed_data_));

    cursor_->Next();
    if (!cursor_->valid())
    {
      LOG(INFO) << "Restarting data fetching from start.";
      cursor_->SeekToFirst();
    }
  }

  /* initialize random seed: */
  srand (time(NULL));

  // read the second half
  for (int item_id = 0; item_id < batch_size / 2; ++item_id)
  {
    int item_label = top_label[item_id];
    int len, index;
    // flip a coin
    if (rand() % 2)
    {
      // select positive example
      len = label_index_set_[item_label].size();
      index = label_index_set_[item_label][rand() % len];
      top_label[item_num + item_id] = item_label;
    }
    else
    {
      // select negative example
      // first select one class to sample from
      int label_to_sample;
      while(1)
      {
        label_to_sample = rand() % label_index_set_.size();
        if (label_to_sample != item_label)
          break;
      }
      len = label_index_set_[label_to_sample].size();
      index = label_index_set_[label_to_sample][rand() % len];      
      top_label[item_num + item_id] = label_to_sample;
    }
    // retrieve the data at index
    string key_str = caffe::format_int(index, 8);
    string value;
    transaction_->Get(key_str, value);
    datum.ParseFromString(value);

    // Apply data transformations (mirror, scale, crop...)
    int offset = data.offset(item_num + item_id);
    transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(transformed_data_));
  }

  // Reshape to loaded data.
  top[0]->ReshapeLike(data);
  // Copy the data
  caffe_copy(data.count(), data.cpu_data(), top[0]->mutable_cpu_data());

  // Reshape to loaded labels.
  top[1]->ReshapeLike(label);
  // Copy the labels.
  caffe_copy(label.count(), label.cpu_data(), top[1]->mutable_cpu_data());
}

INSTANTIATE_CLASS(DataSingletonLayer);
REGISTER_LAYER_CLASS(DataSingleton);

}  // namespace caffe
